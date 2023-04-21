from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from utils.logger import set_color


class AbstractRecommender(nn.Module):


    def __init__(self, config, dataset):
        super(AbstractRecommender, self).__init__()
        self.logger = getLogger()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_USER_ID = config["NEG_PREFIX"] + self.USER_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID

        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        self.USER_SEQ = self.USER_ID + config["LIST_SUFFIX"]
        self.USER_SEQ_LEN = config["USER_LIST_LENGTH_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.max_seq_length = config["MAX_LIST_LENGTH"]

        self.NEG_USER_SEQ = config["NEG_PREFIX"]+self.USER_SEQ
        self.NEG_USER_SEQ_LEN = config["NEG_PREFIX"]+ self.USER_SEQ_LEN
        self.NEG_ITEM_SEQ = config["NEG_PREFIX"]+self.ITEM_SEQ
        self.NEG_ITEM_SEQ_LEN = config["NEG_PREFIX"]+self.ITEM_SEQ_LEN
        # load parameters info
        self.device = config["device"]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def calculate_loss(self, interaction):
        """Calculate the training loss for a batch data.

        Args:
            interaction (dict): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        """Predict the scores between users and items.

        Args:
            interaction (dict): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
                super().__str__()
                + set_color("\nTrainable parameters", "blue")
                + f": {params}"
        )
