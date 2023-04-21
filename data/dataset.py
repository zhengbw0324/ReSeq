
import copy
import os
import pickle

import numpy as np
import pandas as pd
import torch
from logging import getLogger

from utils.utils import ensure_dir
from utils.logger import set_color

class BiSeqRecDataset(torch.utils.data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_name = config["dataset"]
        self.phases = ["train","valid-user","valid-item","test-user","test-item"]
        self.phase = "train"
        self.inter_feats = dict()
        self.logger = getLogger()

        self._get_field_from_config()
        self._load_data()

    def _get_field_from_config(self):
        """Initialization common field names."""
        self.dataset_path = self.config["data_path"]

        self.user_id2token = None
        self.user_token2id = None
        self.item_id2token = None
        self.item_token2id = None
        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.label_field = self.config["LABEL_FIELD"]
        self.time_field = self.config["TIME_FIELD"]
        self.user_id_list_field = self.uid_field + self.config["LIST_SUFFIX"]
        self.item_id_list_field = self.iid_field + self.config["LIST_SUFFIX"]
        self.user_time_list_field = self.uid_field + "_" + self.time_field + self.config["LIST_SUFFIX"]
        self.item_time_list_field = self.iid_field + "_" + self.time_field + self.config["LIST_SUFFIX"]
        self.user_list_length_field = self.config["USER_LIST_LENGTH_FIELD"]
        self.item_list_length_field = self.config["ITEM_LIST_LENGTH_FIELD"]
        self.max_his_list_len = self.config["MAX_LIST_LENGTH"]
        self.neg_users_field = self.config["NEG_USERS_FIELD"]
        self.neg_items_field = self.config["NEG_ITEMS_FIELD"]

        if (self.uid_field is None) ^ (self.iid_field is None):
            raise ValueError(
                "USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time."
            )

    def _load_data(self):

        if not os.path.exists(self.dataset_path):
            raise ValueError(
                "dataset_path not exist."
            )

        for phase in self.phases:
            self.inter_feats[phase] = pd.read_pickle(os.path.join(self.dataset_path, phase+".pkl"))
        self.inter_feat = self.inter_feats[self.phase]
        self.user_his = pd.read_pickle(os.path.join(self.dataset_path, "user_his.pkl"))
        self.item_his = pd.read_pickle(os.path.join(self.dataset_path, "item_his.pkl"))


        with open(os.path.join(self.dataset_path, "user-token.pkl"), "rb") as f:
            user_token = pickle.load(f)
        with open(os.path.join(self.dataset_path, "item-token.pkl"), "rb") as f:
            item_token = pickle.load(f)
        self.user_id2token = user_token["id2token"]
        self.user_token2id = user_token["token2id"]
        self.item_id2token = item_token["id2token"]
        self.item_token2id = item_token["token2id"]
        self.user_num = len(self.user_id2token)
        self.item_num = len(self.item_id2token)

    def set_phase(self, phase):

        if phase not in self.phases:
            raise ValueError(f"Phase [{phase}] not exist.")
        new_dataset = copy.copy(self)
        new_dataset.phase = phase
        if phase != self.phases[0]:
            if "user" in phase:
                new_dataset.neg_list_field = new_dataset.neg_items_field
            else:
                new_dataset.neg_list_field = new_dataset.neg_users_field
        new_dataset.inter_feat = new_dataset.inter_feats[phase]
        return new_dataset

    @property
    def inter_num(self):
        return len(self.inter_feat)

    def __getitem__(self, index):
        if isinstance(index, (list, np.ndarray)):
            df = self.inter_feat.iloc[index]
        elif isinstance(index, torch.Tensor):
            index = index.cpu().numpy()
            df = self.inter_feat.iloc[index]
        else:
            index = [index]
            df = self.inter_feat.iloc[index]
        return df

    def __len__(self):
        return len(self.inter_feat)

    def shuffle(self):
        """Shuffle the interaction records inplace."""
        self.inter_feat.shuffle()

    def sort(self, by, ascending=True):
        self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)

    def save(self):
        """Saving this :class:`Dataset` object to :attr:`config['checkpoint_dir']`."""
        save_dir = self.config["checkpoint_dir"]
        ensure_dir(save_dir)
        file = os.path.join(save_dir, f'{self.config["dataset"]}-dataset.pth')
        self.logger.info(
            set_color("Saving processed dataset into ", "pink") + f"[{file}]"
        )
        with open(file, "wb") as f:
            pickle.dump(self, f)

    def join_his(self, df):
        if self.phase == self.phases[0]:
            df = self.join_user_his(df)
            df = self.join_item_his(df)
        elif "user" in self.phase:
            df = self.join_item_his(df)
        else:
            df = self.join_user_his(df)

        return df

    def join_user_his(self, df):
        df = df.merge(self.user_his, on=self.uid_field, how="left")
        df[self.item_id_list_field] = list(map(self.truncation_his_seq_by_time,
                                               df[self.time_field].values,
                                               df[self.item_id_list_field].values,
                                               df[self.item_time_list_field].values))
        df = df.drop([self.item_time_list_field], axis=1)
        df[self.item_list_length_field] = df[self.item_id_list_field].apply(self.his_len)

        return df

    def join_item_his(self, df):

        df = df.merge(self.item_his, on=self.iid_field, how="left")
        df[self.user_id_list_field] = list(map(self.truncation_his_seq_by_time,
                                          df[self.time_field].values,
                                          df[self.user_id_list_field].values,
                                          df[self.user_time_list_field].values))
        df = df.drop([self.user_time_list_field], axis=1)
        df[self.user_list_length_field] = df[self.user_id_list_field].apply(self.his_len)

        return df

    def his_len(self, value):
        nonzero = value != 0.
        length = nonzero.astype(float).sum()
        return length

    def truncation_his_seq_by_time(self, time, his_seq, time_list):
        index = np.where(time_list < time)[0]
        index = 0 if len(index) == 0 else index[-1]
        new_his_seq = np.array(his_seq[:index + 1])
        if len(new_his_seq) > self.max_his_list_len:
            new_his_seq = new_his_seq[len(new_his_seq) - self.max_his_list_len:]
        return np.concatenate((new_his_seq, np.zeros(self.max_his_list_len - len(new_his_seq))), axis=0)
