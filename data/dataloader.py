from logging import getLogger

import pandas as pd
import torch
import numpy as np
import torch.nn.utils.rnn as rnn_utils

start_iter = False

class AbstractDataLoader(torch.utils.data.DataLoader):

    def __init__(self, config, dataset, shuffle=False):
        self.shuffle = shuffle
        self.config = config
        self._dataset = dataset
        self._init_batch_size_and_step()
        self._batch_size = self.step
        index_sampler = None
        self.generator = torch.Generator()
        self.generator.manual_seed(config["seed"])
        self.logger = getLogger()
        self.data_type = config["data_type"]
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.time_field = dataset.time_field
        self.label_field = dataset.label_field
        self.user_num = self._dataset.user_num
        self.item_num = self._dataset.item_num
        self.sample_size = len(dataset)
        self.neg_prefix = config["NEG_PREFIX"]
        self.list_prefix = config["LIST_SUFFIX"]
        self.max_seq_length = config["MAX_LIST_LENGTH"]
        if self.data_type == "bert":
            self.prepare_bert_data()
        super().__init__(
            dataset=list(range(self.sample_size)),
            batch_size=self.step,
            collate_fn=self.collate_fn,
            num_workers=config["worker"],
            shuffle=shuffle,
            sampler=index_sampler,
            generator=self.generator,
        )

    def _init_batch_size_and_step(self):
        """Initializing :attr:`step` and :attr:`batch_size`."""
        raise NotImplementedError(
            "Method [init_batch_size_and_step] should be implemented"
        )

    def update_config(self, config):

        self.config = config
        self._init_batch_size_and_step()

    def set_batch_size(self, batch_size):

        self._batch_size = batch_size

    def collate_fn(self, index):
        """Collect the sampled index, and apply neg_sampling or other methods to get the final data."""
        raise NotImplementedError("Method [collate_fn] must be implemented.")

    def __iter__(self):
        global start_iter
        start_iter = True
        res = super().__iter__()
        start_iter = False
        return res

    def __getattribute__(self, __name: str):
        global start_iter
        if not start_iter and __name == "dataset":
            __name = "_dataset"
        return super().__getattribute__(__name)

    def transform(self, data):
        new_data = dict()
        for key, value in data.items():
            if self.list_prefix == key[-len(self.list_prefix):]:
                seq_data = [torch.LongTensor(v) for v in value]
                new_data[key] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            else:
                new_data[key] = torch.LongTensor(value)
        return new_data

    def prepare_bert_data(self):
        self.cls = torch.LongTensor([self.user_num + self.item_num - 1])
        self.sep = torch.LongTensor([self.user_num + self.item_num])
        self.USER_SEQ = self.uid_field + self.config["LIST_SUFFIX"]
        self.USER_SEQ_LEN = self.config["USER_LIST_LENGTH_FIELD"]
        self.ITEM_SEQ = self.iid_field + self.config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = self.config["ITEM_LIST_LENGTH_FIELD"]

        self.NEG_USER_SEQ = self.config["NEG_PREFIX"] + self.USER_SEQ
        self.NEG_USER_SEQ_LEN = self.config["NEG_PREFIX"] + self.USER_SEQ_LEN
        self.NEG_ITEM_SEQ = self.config["NEG_PREFIX"] + self.ITEM_SEQ
        self.NEG_ITEM_SEQ_LEN = self.config["NEG_PREFIX"] + self.ITEM_SEQ_LEN

    def transform_seq_data(self, user_seq, user_seq_len, item_seq, item_seq_len):
        new_seqs = []
        seq_type = []
        for useq, useq_len, iseq, iseq_len in zip(user_seq, user_seq_len, item_seq, item_seq_len):
            padding = torch.zeros(2*self.max_seq_length-useq_len-iseq_len, dtype=torch.long)
            new_seq = torch.cat([self.cls, useq[:useq_len],self.sep,
                                 iseq[:iseq_len]+self.user_num-1,self.sep,padding],dim=0)
            type = torch.cat([torch.LongTensor([1]*(useq_len+2)),torch.LongTensor([2]*(iseq_len+1))],dim=0)
            type = torch.cat([type, padding],dim=0)
            new_seqs.append(new_seq)
            seq_type.append(type)
        new_seqs = torch.stack(new_seqs, dim=0)
        seq_type = torch.stack(seq_type, dim=0)
        return new_seqs, seq_type


class BiSeqRecTrainDataloader(AbstractDataLoader):

    def __init__(self, config, dataset, shuffle = False):
        self.neg_sample_num = config["train_neg_sample_num"]
        super(BiSeqRecTrainDataloader, self).__init__(config, dataset, shuffle=shuffle)
        self.user_his = self._dataset.user_his
        self.item_his = self._dataset.item_his
        self.user_time_list_field = self._dataset.user_time_list_field
        self.item_time_list_field = self._dataset.item_time_list_field

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        batch_num = max(batch_size // self.neg_sample_num, 1)
        new_batch_size = batch_num * self.neg_sample_num
        self.step = batch_num
        self.set_batch_size(new_batch_size)

    def update_config(self, config):
        self.neg_sample_num = config["train_neg_sample_num"]
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        neg_data = self._neg_sampling(data)
        data_dict = dict()
        for k in data:
            data_dict[k] = np.repeat(data[k].to_numpy(),self.neg_sample_num)
        for k in neg_data:
            data_dict[self.neg_prefix+k] = neg_data[k].to_numpy()
        data = self.transform(data_dict)
        if self.data_type=="bert":
            data = self.transform_bert_data(data)
        return data

    def _neg_sampling(self, inter_feat):

        user_ids = inter_feat[self.uid_field].to_numpy()
        item_ids = inter_feat[self.iid_field].to_numpy()
        time = inter_feat[self.time_field].to_numpy()
        neg_user_ids, neg_item_ids = self.sample_by_ids(
            user_ids, item_ids, time, self.neg_sample_num
        )
        return self._get_neg_feat(time, neg_user_ids, neg_item_ids)

    def sample_by_ids(self, user_ids, item_ids, time, num):

        user_used_ids = np.array([{i} for i in item_ids])
        item_used_ids = np.array([{i} for i in user_ids])
        neg_item_ids = self.sample(np.arange(len(user_ids)), user_used_ids, time, num, "item")
        neg_user_ids = self.sample(np.arange(len(item_ids)), item_used_ids, time, num, "user")

        return neg_user_ids, neg_item_ids

    def sample(self, key_ids, used_ids, time, num, mode="item"):

        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num

        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        key_ids = np.repeat(key_ids, num)
        time = np.repeat(time, num)
        if mode == "item":
            all = np.arange(1, self.item_num)
            his_time = self.item_his[self.user_time_list_field].to_numpy()
        else:
            all = np.arange(1, self.user_num)
            his_time = self.user_his[self.item_time_list_field].to_numpy()
        remain_num = 0
        step = 0
        while len(check_list) > 0:
            if remain_num == len(check_list):
                step += 1
                if step>100:
                    break
            else:
                step = 0
            remain_num = len(check_list)
            value_ids[check_list] = np.random.choice(all, size=remain_num, replace=True)
            old_check_list = check_list
            check_list = np.array(
                [
                    i
                    for i, used, v, t, tlist in zip(
                    old_check_list,
                    used_ids[key_ids[old_check_list]],
                    value_ids[old_check_list],
                    time[old_check_list],
                    his_time[value_ids[old_check_list]],
                )
                    if v in used or t < tlist[0]
                ]
            )
            for key, v in zip(key_ids[old_check_list], value_ids[old_check_list]):
                used_ids[key].add(v)
        return value_ids

    def _get_neg_feat(self, time, neg_user_ids, neg_item_ids):
        time = time.repeat(self.neg_sample_num)
        neg_feat = pd.DataFrame({self.uid_field: neg_user_ids, self.iid_field: neg_item_ids, self.time_field: time})
        neg_feat = self._dataset.join_his(neg_feat)
        neg_feat = neg_feat.drop([self.time_field], axis=1)
        return neg_feat

    def transform_bert_data(self, data):
        item_seq = data[self.ITEM_SEQ]
        item_seq_len = data[self.ITEM_SEQ_LEN]
        user_seq = data[self.USER_SEQ]
        user_seq_len = data[self.USER_SEQ_LEN]
        neg_item_seq = data[self.NEG_ITEM_SEQ]
        neg_item_seq_len = data[self.NEG_ITEM_SEQ_LEN]
        neg_user_seq = data[self.NEG_USER_SEQ]
        neg_user_seq_len = data[self.NEG_USER_SEQ_LEN]
        pos_seq, pos_seq_type = self.transform_seq_data(user_seq, user_seq_len, item_seq, item_seq_len)
        neg_seq_1, neg_seq_1_type = self.transform_seq_data(user_seq, user_seq_len, neg_item_seq, neg_item_seq_len)
        neg_seq_2, neg_seq_2_type = self.transform_seq_data(neg_user_seq, neg_user_seq_len, item_seq, item_seq_len)

        new_data = dict()
        new_data["pos_seq"] = pos_seq
        new_data["pos_seq_type"] =pos_seq_type
        new_data["neg_seq_1"] =neg_seq_1
        new_data["neg_seq_1_type"] =neg_seq_1_type
        new_data["neg_seq_2"] =neg_seq_2
        new_data["neg_seq_2_type"] =neg_seq_2_type
        return new_data

class BiSeqRecEvalDataloader(AbstractDataLoader):

    def __init__(self, config, dataset, shuffle = False):
        self.neg_num = config["eval_neg_num"]
        if shuffle:
            self.logger.warnning("BiSeqRecEvalDataloader can't shuffle")
            shuffle = False
        super(BiSeqRecEvalDataloader, self).__init__(config, dataset, shuffle=shuffle)
        self.neg_list_field = self._dataset.neg_list_field
        self.phase = self._dataset.phase
        if "user" in self.phase:
            self.candidate_field = self.iid_field
        else:
            self.candidate_field = self.uid_field

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self.times = self.neg_num + 1
        batch_num = max(batch_size // self.times, 1)
        new_batch_size = batch_num * self.times
        self.step = batch_num
        self.set_batch_size(new_batch_size)

    def update_config(self, config):
        self.neg_num = config["eval_neg_num"]
        super().update_config(config)

    def collate_fn(self, index):

        index = np.array(index)
        data = self._dataset[index]
        inter_num = len(data)
        positive_u = np.arange(inter_num)
        positive_i = data[self.candidate_field].to_numpy()
        neg_ids = data[self.neg_list_field].to_numpy()
        data = data.drop([self.neg_list_field], axis=1)
        neg_ids = [neg[:self.neg_num] for neg in neg_ids]
        neg_ids = np.stack(neg_ids, axis=0).T.flatten()
        idx_list = np.tile(positive_u, self.times)

        neg_feat = self._get_neg_feat(data, neg_ids)
        data, all_data = self.merge_data(data, neg_feat)

        data = self.transform(data)
        all_data =  self.transform(all_data)

        if self.data_type=="bert":
            data = self.transform_bert_data(data)
            all_data = self.transform_bert_data(all_data)

        return data, all_data, idx_list, torch.LongTensor(positive_u), torch.LongTensor(positive_i)
    
    def _get_neg_feat(self, df, neg_ids):
        times = np.tile(df[self.time_field].to_numpy(), self.neg_num)
        neg_feat = pd.DataFrame({self.candidate_field: neg_ids, self.time_field: times})
        neg_feat = self._dataset.join_his(neg_feat)
        return neg_feat

    def merge_data(self, pos_data_df, neg_data_df):

        all_data = dict()
        pos_data = dict()
        for k in pos_data_df:
            pos_data[k] = pos_data_df[k].to_numpy()
            if k in neg_data_df.columns:
                all_data[k] = np.concatenate([pos_data_df[k].to_numpy(),neg_data_df[k].to_numpy()], axis=0)
            else:
                all_data[k] = np.tile(pos_data_df[k].to_numpy(), self.times)

        labels = np.zeros(len(pos_data_df)*self.times)
        labels[:len(pos_data_df)] = 1
        all_data[self.label_field] = labels

        return pos_data, all_data

    def transform_bert_data(self, data):
        item_seq = data[self.ITEM_SEQ]
        item_seq_len = data[self.ITEM_SEQ_LEN]
        user_seq = data[self.USER_SEQ]
        user_seq_len = data[self.USER_SEQ_LEN]
        pos_seq, pos_seq_type = self.transform_seq_data(user_seq, user_seq_len, item_seq, item_seq_len)

        new_data = dict()
        new_data["pos_seq"] = pos_seq
        new_data["pos_seq_type"] = pos_seq_type
        new_data[self.candidate_field] = data[self.candidate_field]
        return new_data






