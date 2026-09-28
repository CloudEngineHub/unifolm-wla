import json
import os


def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm

    # Local checkpoint dirs may be named arbitrarily (e.g. a bundled `tokenizer/`
    # folder shipped with a released VLA checkpoint), so the path string alone
    # can't be trusted — check the actual `model_type` from its config.json first.
    model_type = None
    config_json = os.path.join(vlm_name, "config.json") if os.path.isdir(vlm_name) else None
    if config_json and os.path.exists(config_json):
        with open(config_json, "r") as f:
            model_type = json.load(f).get("model_type")

    if model_type == "qwen3_vl" or "Qwen3-VL" in vlm_name or "qwen3_vl" in vlm_name.lower():
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")

