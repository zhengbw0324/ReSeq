import csv
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch

from data_samlpe.model.sasrec import SASRec
import torch.nn.utils.rnn as rnn_utils

csv.field_size_limit(1000 * 1024 * 1024)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',200)
seed = 2023
random.seed(seed)
np.random.seed(seed)

class NegSample():

    def __init__(self, root="./",
                 dataset="ask",
                 view="user"):

        self.root = os.path.join(root, dataset)
        self.dataset = dataset
        self.user_id2token = None
        self.user_token2id = None
        self.job_id2token = None
        self.job_token2id = None
        self.user_num = None
        self.job_num = None
        self.topk = 1000

        self.uid_field = "q_id"
        self.jid_field = "a_id"
        self.label_field = "label"
        self.time_field = "timestamp"
        self.user_id_list_field = self.uid_field + "_list"
        self.job_id_list_field = self.jid_field + "_list"
        self.user_time_list_field = self.uid_field + "_" + self.time_field + "_list"
        self.job_time_list_field = self.jid_field + "_" + self.time_field + "_list"
        self.user_list_length_field = "q_list_length"
        self.job_list_length_field = "a_list_length"
        self.neg_user_list_field = "neg_q"
        self.neg_job_list_field = "neg_a"
        self.max_his_list_len = 50

        self._load_store_data()

        self.veiw = view
        if self.veiw == "user":
            self.neg_list_field = self.neg_job_list_field
            self.all = np.arange(1, self.job_num)
        else:
            self.neg_list_field = self.neg_user_list_field
            self.all = np.arange(1, self.user_num)

        self._neg_sample()


    def _neg_sample(self):
        phases = ["valid", "test"]
        dfs = [self.valid, self.test]

        for phase, df in zip(phases,dfs):
            user_ids = df[self.uid_field].to_numpy()
            job_ids = df[self.jid_field].to_numpy()

            neg_id_list = []
            for uid,jid in zip(user_ids, job_ids):
                used = jid if self.veiw == "user" else uid
                neg_ids = np.random.choice(self.all, size=self.topk, replace=False)
                while used in neg_ids:
                    neg_ids[neg_ids==used] = np.random.choice(self.all, size=1)
                neg_id_list.append(neg_ids)
            df[self.neg_list_field] = neg_id_list
            print(df.head())
            print(df.info())
            df.to_pickle(os.path.join(self.root, phase+"-"+self.veiw+".pkl"))

    def _load_store_data(self):

        self.valid = pd.read_pickle(os.path.join(self.root, "valid.pkl"))
        self.test = pd.read_pickle(os.path.join(self.root, "test.pkl"))
        with open(os.path.join(self.root, "user-token.pkl"), "rb") as f:
            user_token = pickle.load(f)

        with open(os.path.join(self.root, "item-token.pkl"), "rb") as f:
            job_token = pickle.load(f)

        self.user_id2token = user_token["id2token"]
        self.user_token2id = user_token["token2id"]
        self.job_id2token = job_token["id2token"]
        self.job_token2id = job_token["token2id"]

        self.user_num = len(self.user_id2token)
        self.job_num = len(self.job_id2token)


if __name__ == '__main__':
    data = ["ask"]
    view = ["user","item"]
    for d in data:
        print(d)
        for v in view:
            print(v)
            NegSample(dataset=d,view=v)