from .qwen3_vl import Qwen3VL
from .. import MODEL_REGISTRY

MODEL_REGISTRY["qwen3_vl_8b_img"] = {
    "class": Qwen3VL,
    "model_id_mapping": "QWEN3-VL-8B-IMG",
}

# Future capabilities
MODEL_REGISTRY["qwen3_vl_8b_txt"] = {
    "class": Qwen3VL,
    "model_id_mapping": "QWEN3-VL-8B-TXT",
}

MODEL_REGISTRY["qwen3_vl_8b_joint"] = {
    "class": Qwen3VL,
    "model_id_mapping": "QWEN3-VL-8B-JOINT",
}
