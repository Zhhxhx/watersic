from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from quant_layerwise.names import get_hess_name, get_weight_name, concat_dim


def get_bucket_path() -> Path:
    p = os.environ.get("QUANT_BUCKET", None)
    if not p:
        raise RuntimeError(
            "QUANT_BUCKET environment variable is not set. "
            "Example: export QUANT_BUCKET=/home/ubuntu/quant-bucket"
        )
    return Path(p)


# ARGUS 专用 scaffold(BAGEL-und / JanusFlow 抽成 Qwen3-shape)存放在 research 仓库的
# artifacts 目录(在本 submodule 之外)。用 ARGUS_ARTIFACTS_DIR 覆盖以便跨机/跨 checkout;
# 缺省指向原开发机路径。这些值是绝对路径, `base_dir / model_reg[name]` 仍原样返回它们。
_ARGUS_ARTIFACTS = os.environ.get(
    "ARGUS_ARTIFACTS_DIR",
    "/share/liujun/xinhaolee/workspace/research/2027-conf/HPCA/ARGUS/code/artifacts",
)

model_reg: dict[str, str] = {
    "2-7B": "Llama-2-7B",
    "3-70B": "Llama-3-70B-fixed",
    "3-70B-4s": "Llama-3-70B-4shard",
    "2-13B": "Llama-2-13B",
    "3.2-1B": "Llama3.2-1B",
    "3.2-1B-2s": "Llama3.2-1B-2shard",
    "3.2-1B-4s": "Llama3.2-1B-4shard",
    "3-70B-2s": "Llama-3-70B-2shard",
    "3-8B": "Llama-3-8B",
    "qwen3-8B": "Qwen/Qwen3-8B",
    # BAGEL 理解专家(抽成 Qwen3 结构);名字以 qwen3 开头以复用 _is_qwen3/_load_qwen3
    "qwen3-bagelund": os.path.join(_ARGUS_ARTIFACTS, "bagel-und-qwen3shape"),
    # JanusFlow LLM 骨干(DeepSeek-LLM-1.3B, 抽成 Qwen3-shape);同样以 qwen3 开头复用加载路径
    "qwen3-januslm": os.path.join(_ARGUS_ARTIFACTS, "janusflow-llm-qwen3shape"),
}

hess_reg: dict[str, str] = {
    "qwen3-bagelund": "hess_qwen3_bagelund",
    "qwen3-januslm": "hess_qwen3_januslm",
    "2-7B": "hess7",
    "3-70B": "hess3_70",
    "2-13B": "hess7",
    "3-70B-2s": "hess3_70",
    "3.2-1B": "hess1",
    "3.2-1B-2s": "hess1",
    "3-8B": "hess8",
    "qwen3-8B": "hess_qwen3_8",
}

acts_reg: dict[str, str] = {
    "qwen3-bagelund": "acts_qwen3_bagelund",
    "qwen3-januslm": "acts_qwen3_januslm",
    "2-7B": "acts7",
    "3-70B": "acts3_70",
    "2-13B": "acts7",
    "3-70B-2s": "acts3_70",
    "3.2-1B": "acts1",
    "3.2-1B-2s": "acts1",
    "3-8B": "acts8",
    "qwen3-8B": "acts_qwen3_8",
}


def _load_tensor_with_fallbacks(dirpath: Path, stem: str) -> torch.Tensor:
    cands = [dirpath / stem, dirpath / f"{stem}.pt", dirpath / f"{stem}.pth"]
    for p in cands:
        if p.exists():
            return torch.load(p, map_location="cpu")
    raise FileNotFoundError(
        f"Could not find '{stem}' in {dirpath}. Tried: " + ", ".join([c.name for c in cands])
    )


def load_llama_layers(
    model_name: str,
    layers,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    load_acts: bool = False,
    acts_dtype: torch.dtype | None = None,
    acts_root: str | Path | None = None,
):
    base_dir = get_bucket_path()
    model_dir = base_dir / model_reg[model_name]
    hess_dir = base_dir / hess_reg[model_name]

    if load_acts:
        if acts_dtype is None:
            acts_dtype = dtype
        if acts_root is None:
            acts_dir = base_dir / acts_reg[model_name]
        else:
            acts_dir = Path(acts_root)
            if not acts_dir.is_absolute():
                acts_dir = base_dir / acts_dir

    ckpt_paths = list(model_dir.glob("*.pth"))
    print(f"Found {len(ckpt_paths)} checkpoints")
    ckpts: list[dict[str, Any]] = []
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"Loading checkpoint {i}: {ckpt_path.name}")
        ckpts.append(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    return_list = True
    if not isinstance(layers, list):
        return_list = False
        layers = [layers]

    result = []
    for layer_id, weight in layers:
        mod_name = get_hess_name(layer_id, weight)
        weight_name = get_weight_name(layer_id, weight)

        W = torch.cat([ckpt[weight_name] for ckpt in ckpts], dim=concat_dim(weight)).to(device).to(dtype)
        H = _load_tensor_with_fallbacks(hess_dir, mod_name).to(device).to(dtype)

        if load_acts:
            A = _load_tensor_with_fallbacks(acts_dir, mod_name).to(device).to(acts_dtype)
            result.append((W, H, A))
        else:
            result.append((W, H))

    if not return_list:
        return result[0]
    return result
