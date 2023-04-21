from enum import Enum


class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
    - ``VALUE``: Value-based metrics like AUC, etc.
    """

    RANKING = 1
    VALUE = 2


class FeatureType(Enum):
    """Type of features.

    - ``TOKEN``: Token features like user_id and item_id.
    - ``FLOAT``: Float features like rating and timestamp.
    - ``TOKEN_SEQ``: Token sequence features like review.
    - ``FLOAT_SEQ``: Float sequence features like pretrained vector.
    """

    TOKEN = "token"
    FLOAT = "float"
    TOKEN_SEQ = "token_seq"
    FLOAT_SEQ = "float_seq"

