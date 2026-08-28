from __future__ import annotations

DEFAULT_SPECTRAL_PRB_DEPTH = 4
DEFAULT_TFPATCH_PRB_DEPTH = 1

DEFAULT_TBAGNET_OUTPUT_STEM = "tbagnet_sprb4_tfprb1"
DEFAULT_TBAGNET_SHORT_TAG = "s4_t1"
DEFAULT_TBAGNET_LABEL = "TBAGNet_S4_T1"


def apply_default_patch_relation_depths(model_cfg: dict) -> dict:
    if "spectral_relation_depth" not in model_cfg:
        model_cfg["spectral_relation_depth"] = model_cfg.get(
            "spectral_patch_relation_depth",
            DEFAULT_SPECTRAL_PRB_DEPTH,
        )
    if "tfpatch_relation_depth" not in model_cfg:
        model_cfg["tfpatch_relation_depth"] = model_cfg.get(
            "tfpatch_patch_relation_depth",
            DEFAULT_TFPATCH_PRB_DEPTH,
        )
    model_cfg.setdefault("spectral_patch_relation_depth", model_cfg["spectral_relation_depth"])
    model_cfg.setdefault("tfpatch_patch_relation_depth", model_cfg["tfpatch_relation_depth"])
    return model_cfg
