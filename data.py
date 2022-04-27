import paddle
import numpy as np
from tqdm import tqdm

def read_data(angle0_num, n, window_size, shift=1):
    for _ in tqdm(range(angle0_num)):
        angle0 = np.random.rand()
        for example in load_examples(angle0, n, window_size, shift=shift):
            angle_seq, label = example
            angle_seq = np.array(angle_seq).astype(np.float32)
            angle_seq = angle_seq.reshape(window_size, 1)
            label = np.array(label).astype(np.float32)
            yield angle_seq, label

def load_examples(angle0, n, window_size, shift=1):
    angles = gen_data(angle0, n)
    start = 0
    end = window_size
    while end < n:
        angle_seq = angles[start:end]
        label = angles[end]
        yield angle_seq, label
        start += shift
        end += shift
    # print(f"The size of dataset: {len(examples)}")  # n - window_size + 1
    
def gen_data(angle0, n):
    angle = angle0
    angles = []
    for _ in range(n):
        delta = np.random.normal(0, 0.05)
        angle_next = angle + delta
        angle = angle_next
        angles.append(angle)    
    return angles

def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      ):
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

