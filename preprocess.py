import os
import json
import pandas as pd
from paddlenlp.datasets import load_dataset
from data import create_dataloader, read_text_pair, convert_example

project_dir = "/home/th/paddle/question_matching"
data_dir = project_dir + "/data/"

bq_train_path = data_dir + "bq_corpus/train.tsv"
lcqmc_train_path = data_dir + "lcqmc/train.tsv"
paws_train_path = data_dir + "paws-x-zh/train.tsv"

bq_dev_path = data_dir + "bq_corpus/dev.tsv"
lcqmc_dev_path = data_dir + "lcqmc/dev.tsv"
paws_dev_path = data_dir + "paws-x-zh/dev.tsv"

bq_test_path = data_dir + "bq_corpus/test.tsv"
lcqmc_test_path = data_dir + "lcqmc/test.tsv"
paws_test_path = data_dir + "paws-x-zh/test.tsv"

oppo_path = data_dir + "oppp.json"

oppo = json.load(open(oppo_path))
oppo_train = oppo["train"]
oppo_dev = oppo["dev"]
oppo_test = oppo["test"]

def to_txt(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            q1 = d["q1"]
            q2 = d["q2"]
            label = d["label"]
            s = q1 + "\t" + q2 + "\t" + label + "\n"
            f.write(s)

oppo_dir = data_dir + "oppo"

# to_txt(oppo_train, oppo_dir + "/train.txt")
# to_txt(oppo_dev, oppo_dir + "/dev.txt")
# to_txt(oppo_dev, oppo_dir + "/test.txt")





test_ds = load_dataset(
        read_text_pair, data_path=data_dir + "test_0.tsv", is_test=True, lazy=False)
print(len(test_ds))
