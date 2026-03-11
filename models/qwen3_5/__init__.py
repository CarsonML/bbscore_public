from .qwen3_5 import Qwen3_5
from .. import MODEL_REGISTRY

MODEL_REGISTRY["qwen3_5_9b_txt"] = {
    "class": Qwen3_5,
    "model_id_mapping": "QWEN3_5-9B-TXT",
}

MODEL_REGISTRY["qwen3_5_9b_img"] = {
    "class": Qwen3_5,
    "model_id_mapping": "QWEN3_5-9B-IMG",
}

MODEL_REGISTRY["qwen3_5_9b_joint"] = {
    "class": Qwen3_5,
    "model_id_mapping": "QWEN3_5-9B-JOINT",
}
