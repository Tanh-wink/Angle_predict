from cmath import inf
from functools import partial
import argparse
import os
import random
import time
import logging

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddle.nn import loss

from data import create_dataloader, read_data
from model import AnglePredict

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def getArgs():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default='/home/th/paddle/angle_predict/checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
    parser.add_argument("--train_batch_size", default=256, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_step", default=500, type=int, help="Step interval for evaluation.")
    parser.add_argument("--log_step", default=100, type=int, help="Step interval for logging and printing loss.")
    parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
    parser.add_argument("--warmup_proportion", default=1000, type=int, help="Linear warmup proption over the training process.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument('--union', choices=['logits', 'co_logits'], default="co_logits", help="Choice which logits to select answer.")
    parser.add_argument("--gpuids", type=str, default="1", required=False, help="set gpu ids which use to perform")
    
    parser.add_argument("--train_angle0_num", default=100000, type=int, help="The nume of angles begin with per angle0")
    parser.add_argument("--dev_angle0_num", default=1000, type=int, help="The nume of angles begin with per angle0")
    parser.add_argument("--angle_num", default=100, type=int, help="The nume of angles begin with per angle0")
    parser.add_argument("--shift", default=1, type=int, help="step for generate angle sequences")
    parser.add_argument("--window_size", default=50, type=int, help="window size for angle input sequences")
    # rnn parameters 
    parser.add_argument("--input_size", type=int, default=1, required=False, help="input size of RNN layer one")
    parser.add_argument("--hidden_size", type=int, default=100, required=False, help="hidden size of RNN layer one")
    parser.add_argument("--num_layers", type=int, default=2, required=False, help="select which rnn style for training")
    parser.add_argument("--rnn_style", choices=['gru', 'lstm'], default="gru", required=False, help="select which rnn style for training")
    
    args = parser.parse_args()
    return args
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    losses = []
    total_num = 0
    start = time.time()
    errors = []
    for batch in data_loader:
        angle_seq, labels = batch
        total_num += len(labels)
        logits = model(angle_seq=angle_seq)
        error = paddle.abs(labels - logits)
        errors += error.tolist()
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
    errors_rate = sum(errors) / len(errors)
    cost_times = time.time()-start
    logger.info("eval_loss: {:.4}, errors_rate: {:.4}, cost_times:{:.4} s, eval_speed: {:.4} item/ms".format(
        np.mean(losses), errors_rate, cost_times, (cost_times / total_num) * 1000))
    model.train()
    return errors_rate


def train(args):
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args.seed)
    train_ds = load_dataset(read_data, angle0_num=args.train_angle0_num, n=args.angle_num, 
                            window_size=args.window_size, shift=args.shift, lazy=False)

    dev_ds = load_dataset(read_data, angle0_num=args.dev_angle0_num, n=args.angle_num, 
                            window_size=args.window_size, shift=args.shift, lazy=False)
    
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.train_batch_size,
        batchify_fn=None)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=None)

    model = AnglePredict(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers
                               , rnn_style=args.rnn_style)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    logger.info(f"The number of examples in train set: {len(train_ds)}")
    logger.info(f"The number of examples in dev set: {len(dev_ds)}")
    logger.info(f"All training steps: {num_training_steps}")

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = nn.loss.MSELoss()
    global_step = 0
    best_errors_rate = inf
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        
        for step, batch in enumerate(train_data_loader, start=1):
            
            angle_seq, labels = batch
            logits = model(angle_seq=angle_seq)
            error = paddle.abs(labels - logits)
            loss = criterion(logits, labels)
            global_step += 1
            if global_step % args.log_step == 0 and rank == 0:
                logger.info(
                    "global step %d, epoch: %d, loss: %.4f, estim_error: %.4f, speed: %.2f step/s"
                    % (global_step, epoch, loss, error.mean(), args.log_step / (time.time() - tic_train)))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0:
                
                errors_rate = evaluate(model, criterion, dev_data_loader)
                if errors_rate < best_errors_rate:
                    save_dir = os.path.join(args.save_dir,
                                            "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    logger.info(f"****({best_errors_rate}------->{errors_rate})****")
                    logger.info(f"Saved the best model at {save_param_path}")
                    best_errors_rate = errors_rate

            if global_step == args.max_steps:
                return

def run():
    args = getArgs()
    
    train(args)


if __name__ == "__main__":
    
    run()