from functools import partial
import argparse
import sys
import os
import random
import time
from tqdm import tqdm
import logging
import pandas as pd

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from data import create_dataloader, read_text_pair, convert_example, convert_example_pair
from model import QuestionMatching, QuestionMatching_2stage

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="/home/th/paddle/question_matching/data/test.tsv", help="The full path of input file")
parser.add_argument("--result_file", type=str, default="/home/th/paddle/question_matching/data/predict_result.csv", help="The result file name")
parser.add_argument("--params_path", type=str, default="/home/th/paddle/question_matching/checkpoints/model_29400_94.5_78.3/model_state.pdparams", help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--union', choices=['logits', 'co_logits'], default="co_logits", help="Choice which logits to select answer.")
parser.add_argument("--gpuids", type=str, default="2", required=False, help="set gpu ids which use to perform")
parser.add_argument("--rdrop_coef", default=0.1, type=float, help="The coefficient of"
    "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
    
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    """
    Predicts the data labels.
    Args:
        model (obj:`QuestionMatching`): A model to calculate whether the question pair is semantic similar or not.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    batch_logits = []

    model.eval()

    with paddle.no_grad():
        for batch_data in tqdm(data_loader):
            input_ids, token_type_ids, input_ids2, token_type_ids2 = batch_data

            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            outputs = model( input_ids=input_ids, token_type_ids=token_type_ids, input_ids2=input_ids2,token_type_ids2=token_type_ids2)
            if args.rdrop_coef > 0 :
                batch_logit, _, co_logits = outputs
                if args.union == "co_logits":
                    batch_logit = co_logits
            else:
                batch_logit, _ = outputs

            batch_logits.append(batch_logit.numpy())

        batch_logits = np.concatenate(batch_logits, axis=0)

        return batch_logits


if __name__ == "__main__":
    paddle.set_device(args.device)

    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained( 'ernie-gram-zh')
    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained( 'ernie-gram-zh')

    trans_func = partial(
        convert_example_pair,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids2
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids2
    ): [data for data in fn(samples)]

    test_ds = load_dataset(
        read_text_pair, data_path=args.input_file, is_test=True, lazy=False)
    logger.info(f"test set size:{len(test_ds)}")
    test_data_loader = create_dataloader(
        test_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    model = QuestionMatching_2stage(pretrained_model, rdrop_coef=args.rdrop_coef)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    y_probs = predict(model, test_data_loader)
    y_preds = np.argmax(y_probs, axis=1)
    test_results = []
    with open(args.result_file, 'w', encoding="utf-8") as f:
        for y_pred in y_preds:
            f.write(str(y_pred) + "\n")
    logger.info(f"saved predict result at {args.result_file}")