import os
import sys
import base64
import io
import tempfile
from typing import Any, Dict, Optional, List

import numpy as np
from PIL import Image

import runpod

# Make notebook inference importable (matches repo usage)
# The SAM3D repo keeps inference.py under "notebook"
sys.path.append("/workspace/sam-3d-objects/notebook")
from inference import Inference  # type: ignore  # noqa: E402


# -----------------------------
# Warm model load (one per worker)
# -----------------------------
TAG = os.environ.get("SAM3D_TAG", "hf")
CONFIG_PATH = os.environ.get("SAM3D_CONFIG", f"checkpoints/{TAG}/pipeline.yaml")

_infer: Optional[Inference] = None


def get_infer() -> Inference:
    global _infer
    if _infer is None:
        # compile=False is safer for serverless stability
        _infer = Inference(CONFIG_PATH, compile=False)
    return _infer


# -----------------------------
# Helpers: base64 <-> arrays/files
# -----------------------------
def b64_to_rgb_np(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def b64_to_mask_np(b64_str: str, width: int, height: int) -> np.ndarray:
    """
    Frontend sends a PNG mask (often RGBA with alpha paint).
    Convert to binary (0/1) and resize to match image.
    """
    raw = base64.b64decode(b64_str)
    m = Image.open(io.BytesIO(raw))

    if m.mode == "RGBA":
        # Use alpha channel as mask
        alpha = m.split()[-1]
        m = alpha
    else:
        # Use luminance
        m = m.convert("L")

    m = m.resize((width, height), resample=Image.NEAREST)
    arr = np.array(m, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_outputs_list(options: Any) -> List[str]:
    # options may be dict or None
    if not isinstance(options, dict):
        return ["ply"]
    out = options.get("output", ["ply"])
    if isinstance(out, list):
        return [str(x).lower() for x in out]
    # allow "ply,glb" string etc.
    if isinstance(out, str):
        return [x.strip().lower() for x in out.split(",") if x.strip()]
    return ["ply"]


# -----------------------------
# RunPod handler
# -----------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod job payload:
      job = { "id": "...", "input": { ... } }
    """
    inp = job.get("input", {}) if isinstance(job, dict) else {}

    image_b64 = inp.get("imageBase64")
    mask_b64 = inp.get("maskBase64")
    options = inp.get("options", {})

    if not image_b64 or not mask_b64:
        return {"error": "imageBase64 and maskBase64 are required"}

    seed = int(options.get("seed", 42)) if isinstance(options, dict) else 42
    outputs = _get_outputs_list(options)

    # Decode input
    image = b64_to_rgb_np(image_b64)
    mask = b64_to_mask_np(mask_b64, image.shape[1], image.shape[0])

    # Run inference
    infer = get_infer()
    out = infer(image, mask, seed=seed)

    result: Dict[str, Any] = {"meta": {"seed": seed}}

    # Export requested formats
    with tempfile.TemporaryDirectory() as td:
        # PLY via gaussian splat
        if "ply" in outputs:
            ply_path = os.path.join(td, "out.ply")
            if isinstance(out, dict) and "gs" in out and hasattr(out["gs"], "save_ply"):
                out["gs"].save_ply(ply_path)
                result["plyBase64"] = file_to_b64(ply_path)
            else:
                return {"error": "PLY requested but out['gs'].save_ply not available"}

        # GLB via mesh export (try common method names)
        if "glb" in outputs:
            glb_path = os.path.join(td, "out.glb")
            mesh_obj = out.get("mesh") if isinstance(out, dict) else None
            if mesh_obj is None:
                return {"error": "GLB requested but out['mesh'] is missing"}

            if hasattr(mesh_obj, "to_glb"):
                mesh_obj.to_glb(glb_path)
            elif hasattr(mesh_obj, "save_glb"):
                mesh_obj.save_glb(glb_path)
            elif hasattr(mesh_obj, "export"):
                mesh_obj.export(glb_path)
            else:
                return {"error": "GLB requested but mesh export method not found on out['mesh']"}

            result["glbBase64"] = file_to_b64(glb_path)

    return result


runpod.serverless.start({"handler": handler})
