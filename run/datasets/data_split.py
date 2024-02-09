
import csv
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch

csv.field_size_limit(500 * 1024 * 1024)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',200)

seed = 2023
random.seed(seed)
np.random.seed(seed)

class DataSplit():

    def __init__(self, root = './', dataset="ask"):
        self.root = os.path.join(root,dataset)
        self.dataset = dataset
        self.inter_file = dataset+'.inter'
        self.user_token = "user-token.pkl"
        self.job_token = "item-token.pkl"

        self.user_id2token = None
        self.user_token2id = None
        self.job_id2token = None
        self.job_token2id = None
        self.user_num = None
        self.job_num = None

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
        self.max_his_list_len = 50

        self.inter = self._load_feat(os.path.join(self.root, self.inter_file))
        # print(self.inter[self.time_field].max(), self.inter[self.time_field].min())
        self._remap_ID_all()
        self._data_processing()
        print(self.inter.info())
        self.inter = self.inter.drop(['direct', self.user_time_list_field, self.job_time_list_field], axis=1)

        self._split_data(valid_time=1404144000, test_time=1430409600) #for ask
        # self._split_data(valid_time=1454256000, test_time=1455638400) #for stackoverflow
        print(self.train.head())
        print(self.train.info())
        print(self.valid.head())
        print(self.valid.info())
        print(self.test.head())
        print(self.test.info())
        print(self.user_his.head())
        print(self.user_his.info())
        print(self.job_his.head())
        print(self.job_his.info())
        self._store_data()



    def _load_feat(self, filepath):

        field_separator = "\t"
        columns = []
        usecols = []
        dtype = {}
        encoding = "utf-8"
        with open(filepath, "r", encoding=encoding) as f:
            head = f.readline().strip()
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(":")
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype=="float" else str

        df = pd.read_csv(
            filepath,
            delimiter=field_separator,
            usecols=usecols,
            dtype=dtype,
            encoding=encoding,
            engine="python",
        )
        df.columns = columns

        return df

    def _remap_ID_all(self):

        with open(os.path.join(self.root, self.user_token), "rb") as f:
            user_token = pickle.load(f)

        with open(os.path.join(self.root, self.job_token), "rb") as f:
            job_token = pickle.load(f)

        self.user_id2token = user_token["id2token"]
        self.user_token2id = user_token["token2id"]
        self.job_id2token = job_token["id2token"]
        self.job_token2id = job_token["token2id"]

        self.user_num = len(self.user_id2token)
        self.job_num = len(self.job_id2token)
        # print(self.inter.head())
        print(self.user_num, self.job_num)
        self.inter[self.uid_field] = self.inter[self.uid_field].map(self.user_token2id)
        self.inter[self.jid_field] = self.inter[self.jid_field].map(self.job_token2id)
        # print(self.inter.head())
        # print(self.user_token2id)
        # print(self.job_token2id)
        
    def _data_processing(self):

        self._get_inter_his()

        self.inter = self.inter.merge(self.user_his, on=self.uid_field)
        self.inter = self.inter.merge(self.job_his, on=self.jid_field)

        def get_user_his_seq(row):
            iid = row[self.jid_field]
            user_his = row[self.job_id_list_field]
            user_his_seq = np.array(user_his[:user_his.index(iid)])
            return user_his_seq

        def get_job_his_seq(row):
            uid = row[self.uid_field]
            job_his = row[self.user_id_list_field]
            job_his_seq = np.array(job_his[:job_his.index(uid)])
            return job_his_seq

        self.inter[self.job_id_list_field] = self.inter.apply(get_user_his_seq, axis=1)
        self.inter[self.user_id_list_field] = self.inter.apply(get_job_his_seq, axis=1)
        # self.inter[self.job_time_list_field] = self.inter.apply(self.get_user_his_time_seq, axis=1)
        # self.inter[self.user_time_list_field] = self.inter.apply(self.get_job_his_time_seq, axis=1)
        self.inter[self.job_id_list_field] = self.inter[self.job_id_list_field].apply(self.padding)
        self.inter[self.user_id_list_field] = self.inter[self.user_id_list_field].apply(self.padding)
        # self.inter[self.job_time_list_field] = self.inter[self.job_time_list_field].apply(self.padding)
        # self.inter[self.user_time_list_field] = self.inter[self.user_time_list_field].apply(self.padding)
        self.inter[self.job_list_length_field] = self.inter[self.job_id_list_field].apply(self.his_len)
        self.inter[self.user_list_length_field] = self.inter[self.user_id_list_field].apply(self.his_len)

        self._his_processing()

        self.inter = self.inter[(self.inter[self.job_list_length_field] > 0)
                                & (self.inter[self.user_list_length_field] > 0)]
        self.inter.reset_index(drop=True, inplace=True)

    def _get_inter_his(self):

        df_inter = pd.DataFrame({self.uid_field: self.inter[self.uid_field],
                                      self.jid_field: self.inter[self.jid_field],
                                      self.time_field: self.inter[self.time_field]})

        df_inter.sort_values(by=[self.time_field, self.uid_field, self.jid_field], ascending=True, inplace=True)

        f = lambda x: x.to_list()

        self.user_his = df_inter.groupby(self.uid_field).agg(
            {self.jid_field: f,
             self.time_field: f})
        self.user_his.reset_index(inplace=True)
        self.user_his.columns = [self.uid_field, self.job_id_list_field, self.job_time_list_field]

        self.job_his = df_inter.groupby(self.jid_field).agg(
            {self.uid_field: f,
             self.time_field: f})
        self.job_his.reset_index(inplace=True)
        self.job_his.columns = [self.jid_field, self.user_id_list_field, self.user_time_list_field]

    def _his_processing(self):

        def fill_nan(value):
            if isinstance(value, (list, np.ndarray, torch.Tensor)):
                return value
            else:
                return []

        self.user_his[self.job_list_length_field] = self.user_his[self.job_id_list_field].apply(len)
        self.job_his[self.user_list_length_field] = self.job_his[self.user_id_list_field].apply(len)

        new_user_his_df = pd.DataFrame({self.uid_field: np.arange(self.user_num)})
        self.user_his = pd.merge(new_user_his_df, self.user_his, on=self.uid_field, how='left')
        self.user_his.fillna(value=0, inplace=True)
        self.user_his[self.job_id_list_field] = self.user_his[self.job_id_list_field].apply(fill_nan)
        self.user_his[self.job_time_list_field] = self.user_his[self.job_time_list_field].apply(fill_nan)

        new_job_his_df = pd.DataFrame({self.jid_field: np.arange(self.job_num)})
        self.job_his = pd.merge(new_job_his_df, self.job_his, on=self.jid_field, how='left')
        self.job_his.fillna(value=0, inplace=True)
        self.job_his[self.user_id_list_field] = self.job_his[self.user_id_list_field].apply(fill_nan)
        self.job_his[self.user_time_list_field] = self.job_his[self.user_time_list_field].apply(fill_nan)

    def _split_data(self, valid_time=1561910400, test_time=1562515200):

        self.train = self.inter[self.inter[self.time_field] < valid_time]
        self.valid_test = self.inter[self.inter[self.time_field] >= valid_time]
        self.valid = self.valid_test[self.valid_test[self.time_field] < test_time]
        self.test = self.valid_test[self.valid_test[self.time_field] >= test_time]
        self.train.reset_index(drop=True, inplace=True)
        self.valid.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

    def _store_data(self):

        self.train.to_pickle(os.path.join(self.root, "train.pkl"))
        self.valid.to_pickle(os.path.join(self.root, "valid.pkl"))
        self.test.to_pickle(os.path.join(self.root, "test.pkl"))
        self.user_his.to_pickle(os.path.join(self.root, "user_his.pkl"))
        self.job_his.to_pickle(os.path.join(self.root, "item_his.pkl"))


    def get_user_his_time_seq(self, row):
        seq_len = len(row[self.job_id_list_field])
        user_his_time_list = row[self.job_time_list_field]
        user_his_time_list = np.array(user_his_time_list[:seq_len])
        return user_his_time_list

    def get_job_his_time_seq(self, row):
        seq_len = len(row[self.user_id_list_field])
        job_his_time_list = row[self.user_time_list_field]
        job_his_time_list = np.array(job_his_time_list[:seq_len])
        return job_his_time_list

    def his_len(self, value):
        nonzero = value != 0.
        length = nonzero.astype(float).sum()
        return length

    def padding(self, x):
        if len(x) > self.max_his_list_len:
            x = x[len(x) - self.max_his_list_len:]
        return np.concatenate((x, np.zeros(self.max_his_list_len - len(x))), axis=0)

if __name__ =='__main__':
    data = ["ask","stackoverflow"]
    for d in data:
        print(d)
        DataSplit(dataset=d)

