import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.abstract import AbstractRecommender
from model.layers import Transformer
from model.loss import BPRLoss

class ReSeq(AbstractRecommender):

    def __init__(self, config, dataset):
        super(ReSeq, self).__init__(config, dataset)

        # load parameters info
        self.n_factors = config["n_factors"]
        self.embedding_size = config["embedding_size"]
        self.initializer_range = config["initializer_range"]
        self.kd_loss_weight = config['kd_loss_weight']
        self.t_weight = config["t_weight"]
        self.temperature = config['temperature']
        self.max_seq_length = config["MAX_LIST_LENGTH"]
        self.detach_t = config["detach_t"]
        self.T_weight = nn.Parameter(torch.normal(mean=0.0, std=self.initializer_range, size=(self.max_seq_length,)),
                                   requires_grad=True)

        # define layers and loss
        self.user_f_embedding = nn.Embedding(self.n_users + 1, self.n_factors, padding_idx=0)
        self.item_f_embedding = nn.Embedding(self.n_items + 1, self.n_factors, padding_idx=0)
        self.user_p_embedding = nn.Embedding(self.n_users + 1, self.n_factors, padding_idx=0)
        self.item_p_embedding = nn.Embedding(self.n_items + 1, self.n_factors, padding_idx=0)

        self.up_if_space = nn.Parameter(torch.randn(self.n_factors, self.embedding_size), requires_grad=True)
        self.ip_uf_space = nn.Parameter(torch.randn(self.n_factors, self.embedding_size), requires_grad=True)

        self.user_preference = Transformer(config, bidirectional=False)
        self.item_preference = Transformer(config, bidirectional=False)
        self.user_feature = Transformer(config, bidirectional=True)
        self.item_feature = Transformer(config, bidirectional=True)

        self.rec_loss = BPRLoss()
        self.kd_loss = nn.MSELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, user_seq, item_seq, users=None, items=None):

        ucls = torch.full((user_seq.size(0), 1), self.n_users).to(self.device)
        icls = torch.full((item_seq.size(0), 1), self.n_items).to(self.device)
        cls_user_seq = torch.cat([ucls, user_seq], dim=1)
        cls_item_seq = torch.cat([icls, item_seq], dim=1)

        user_f_emb = self.user_f_embedding(cls_user_seq)
        item_f_emb = self.item_f_embedding(cls_item_seq)
        user_p_emb = self.user_p_embedding(cls_user_seq)
        item_p_emb = self.item_p_embedding(cls_item_seq)

        user_p_emb = torch.matmul(user_p_emb, self.up_if_space)
        item_f_emb = torch.matmul(item_f_emb, self.up_if_space)
        user_f_emb = torch.matmul(user_f_emb, self.ip_uf_space)
        item_p_emb = torch.matmul(item_p_emb, self.ip_uf_space)

        user_p = self.user_preference(cls_item_seq, item_f_emb)
        item_p = self.item_preference(cls_user_seq, user_f_emb)
        user_f = self.user_feature(cls_item_seq, item_p_emb)
        item_f = self.item_feature(cls_user_seq, user_p_emb)

        cls_up = user_p[:, 0]
        cls_ip = item_p[:, 0]
        user_p = user_p[:, 1:]
        item_p = item_p[:, 1:]

        cls_uf = user_f[:, 0]
        cls_if = item_f[:, 0]
        user_f = user_f[:, 1:]
        item_f = item_f[:, 1:]

        user_seq_mask = (user_seq != 0).float()  # [B, user_seq_len]
        item_seq_mask = (item_seq != 0).float()  # [B, item_seq_len]

        user_origin_f, item_origin_f, user_origin_p, item_origin_p= None, None, None, None
        if users is not None and items is not None:
            user_origin_f = self.user_f_embedding(users)
            item_origin_f = self.item_f_embedding(items)
            user_origin_p = self.user_p_embedding(users)
            item_origin_p = self.item_p_embedding(items)

            user_origin_p = torch.matmul(user_origin_p, self.up_if_space)
            item_origin_f = torch.matmul(item_origin_f, self.up_if_space)
            user_origin_f = torch.matmul(user_origin_f, self.ip_uf_space)
            item_origin_p = torch.matmul(item_origin_p, self.ip_uf_space)

        user_p = user_p * item_seq_mask.unsqueeze(-1).expand_as(user_p)
        user_f = user_f * item_seq_mask.unsqueeze(-1).expand_as(user_f)
        item_p = item_p * user_seq_mask.unsqueeze(-1).expand_as(item_p)
        item_f = item_f * user_seq_mask.unsqueeze(-1).expand_as(item_f)

        return (cls_up, user_p, user_origin_p, item_seq_mask), (cls_uf, user_f, user_origin_f, item_seq_mask), \
               (cls_ip, item_p, item_origin_p, user_seq_mask), (cls_if, item_f, item_origin_f, user_seq_mask)

    def calculate_loss(self, interaction):

        users = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]

        items = interaction[self.ITEM_ID]
        user_seq = interaction[self.USER_SEQ]

        neg_users = interaction[self.NEG_USER_ID]
        neg_item_seq = interaction[self.NEG_ITEM_SEQ]

        neg_items = interaction[self.NEG_ITEM_ID]
        neg_user_seq = interaction[self.NEG_USER_SEQ]

        user_p_output, user_f_output, item_p_output, item_f_output = self.forward(user_seq, item_seq, users, items)

        neg_user_p_output, neg_user_f_output, neg_item_p_output, neg_item_f_output = self.forward(neg_user_seq, neg_item_seq, neg_users, neg_items)

        pos_score_s, pos_score_t = self.get_score(user_p_output, user_f_output, item_p_output, item_f_output)
        neg1_score_s, neg1_score_t = self.get_score(neg_user_p_output, user_f_output, item_p_output,
                                                                item_f_output)
        neg2_score_s, neg2_score_t = self.get_score(user_p_output, neg_user_f_output, item_p_output,
                                                                item_f_output)
        neg3_score_s, neg3_score_t = self.get_score(user_p_output, user_f_output, neg_item_p_output,
                                                                item_f_output)
        neg4_score_s, neg4_score_t = self.get_score(user_p_output, user_f_output, item_p_output,
                                                                neg_item_f_output)

        rec_loss_s = self.rec_loss(pos_score_s, neg1_score_s) + self.rec_loss(pos_score_s, neg2_score_s) + \
                     self.rec_loss(pos_score_s, neg3_score_s) + self.rec_loss(pos_score_s, neg4_score_s)
        rec_loss_t = self.rec_loss(pos_score_t, neg1_score_t) + self.rec_loss(pos_score_t, neg2_score_t) + \
                     self.rec_loss(pos_score_t, neg3_score_t) + self.rec_loss(pos_score_t, neg4_score_t)


        loss = rec_loss_s + self.t_weight * rec_loss_t

        if self.kd_loss_weight is not None and self.kd_loss_weight > 0:
            kd_loss = self.kd_loss((pos_score_s - neg1_score_s), (pos_score_t - neg1_score_t).detach()) + \
                      self.kd_loss((pos_score_s - neg2_score_s), (pos_score_t - neg2_score_t).detach()) + \
                      self.kd_loss((pos_score_s - neg3_score_s), (pos_score_t - neg3_score_t).detach()) + \
                      self.kd_loss((pos_score_s - neg4_score_s), (pos_score_t - neg4_score_t).detach())
            loss += self.kd_loss_weight * kd_loss

        return loss

    def get_score(self, user_p_output, user_f_output, item_p_output, item_f_output, training=True):
        cls_up, user_p, u_origin_p, up_mask = user_p_output
        cls_uf, user_f, u_origin_f, uf_mask = user_f_output
        cls_ip, item_p, i_origin_p, ip_mask = item_p_output
        cls_if, item_f, i_origin_f, if_mask = item_f_output

        score_1 = torch.sum(cls_up * cls_if, dim=-1)
        score_2 = torch.sum(cls_ip * cls_uf, dim=-1)

        if not training:
            return score_1 + score_2

        up_w = torch.matmul(user_p, i_origin_f.unsqueeze(-1)).squeeze() * up_mask
        uf_w = torch.matmul(user_f, i_origin_p.unsqueeze(-1)).squeeze() * uf_mask
        ip_w = torch.matmul(item_p, u_origin_f.unsqueeze(-1)).squeeze() * ip_mask
        if_w = torch.matmul(item_f, u_origin_p.unsqueeze(-1)).squeeze() * if_mask

        score_3 = user_p @ item_f.permute(0, 2, 1)
        score_4 = item_p @ user_f.permute(0, 2, 1)

        ut_w = self.get_weight(up_mask.sum(dim=1).long(), up_mask)
        it_w = self.get_weight(ip_mask.sum(dim=1).long(), ip_mask)

        up_w = up_w + ut_w
        ip_w = ip_w + it_w
        uf_w[~(uf_mask.bool())] = -100000
        if_w[~(if_mask.bool())] = -100000

        up_w = F.softmax(up_w, dim=-1)
        ip_w = F.softmax(ip_w, dim=-1)
        uf_w = F.softmax(uf_w, dim=-1).unsqueeze(1)
        if_w = F.softmax(if_w, dim=-1).unsqueeze(1)

        score_3 = torch.sum(score_3 * if_w, dim=-1)
        score_4 = torch.sum(score_4 * uf_w, dim=-1)

        score_3 = torch.sum(score_3 * up_w, dim=-1)
        score_4 = torch.sum(score_4 * ip_w, dim=-1)

        return score_1 + score_2, score_3 + score_4

    def get_weight(self, seq_len, mask):

        time_ids = torch.arange(mask.size(1), dtype=torch.long, device=mask.device)
        relative_time_ids = seq_len.unsqueeze(-1).expand_as(mask) - time_ids.unsqueeze(0).expand_as(mask)
        w = self.T_weight[relative_time_ids - 1]
        w[~(mask.bool())] = -100000

        return w

    def predict(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        user_seq = interaction[self.USER_SEQ]
        user_p_output, user_f_output, item_p_output, item_f_output = self.forward(user_seq, item_seq)
        scores = self.get_score(user_p_output, user_f_output, item_p_output, item_f_output, training=False)

        return scores

    def neg_sample_predict(self, interaction, all_interaction, item_field):

        if item_field == self.ITEM_ID:
            item_seq = interaction[self.ITEM_SEQ]
            user_seq = all_interaction[self.USER_SEQ]
            times = user_seq.size(0) // item_seq.size(0)

            user_p_output, user_f_output, item_p_output, item_f_output = self.forward(user_seq, item_seq)
            user_p_output = (user_p_output[0].repeat(times, 1),
                             user_p_output[1].repeat(times, 1, 1),
                             None,
                             user_p_output[3].repeat(times, 1))
            user_f_output = (user_f_output[0].repeat(times, 1),
                             user_f_output[1].repeat(times, 1, 1),
                             None,
                             user_f_output[3].repeat(times, 1))
        else:
            user_seq = interaction[self.USER_SEQ]
            item_seq = all_interaction[self.ITEM_SEQ]
            times = item_seq.size(0) // user_seq.size(0)

            user_p_output, user_f_output, item_p_output, item_f_output = self.forward(user_seq, item_seq)
            item_p_output = (item_p_output[0].repeat(times, 1),
                             item_p_output[1].repeat(times, 1, 1),
                             None,
                             item_p_output[3].repeat(times, 1))
            item_f_output = (item_f_output[0].repeat(times, 1),
                             item_f_output[1].repeat(times, 1, 1),
                             None,
                             item_f_output[3].repeat(times, 1))

        scores = self.get_score(user_p_output, user_f_output, item_p_output, item_f_output, training=False)

        return scores.view(-1)


