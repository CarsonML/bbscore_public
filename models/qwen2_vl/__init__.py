from .qwen2_vl import Qwen2VL
from .. import MODEL_REGISTRY

MODEL_REGISTRY["qwen2_vl_2b_img"] = {
    "class": Qwen2VL,
    "model_id_mapping": "QWEN2-VL-2B-IMG",
}

# Future capabilities
MODEL_REGISTRY["qwen2_vl_2b_txt"] = {
    "class": Qwen2VL,
    "model_id_mapping": "QWEN2-VL-2B-TXT",
}

MODEL_REGISTRY["qwen2_vl_2b_joint"] = {
    "class": Qwen2VL,
    "model_id_mapping": "QWEN2-VL-2B-JOINT",
}
