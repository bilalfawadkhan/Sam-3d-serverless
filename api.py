import base64
import io
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Import SAM3D inference exactly like the official quickstart ---
# They do:
#   sys.path.append("notebook")
#   from inference import Inference, load_image, load_single_mask
#   inference = Inference(config_path, compile=False)
#   output = inference(image, mask, seed=42)
# :contentReference[oaicite:2]{index=2}
sys.path.append("notebook")
from inference import Inference  # type: ignore


# ----------------------------
# Request / Response schemas
# ----------------------------
class Options(BaseModel):
    output: List[str] = Field(default_factory=lambda: ["ply", "glb"])
    seed: int = 42
    compile: bool = False  # keep false unless you know it's stable on your GPU/runtime


class Sam3DRequest(BaseModel):
    imageBase64: str
    maskBase64: str
    options: Options = Field(default_factory=Options)


class Sam3DResponse(BaseModel):
    glbBase64: Optional[str] = None
    plyBase64: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# Helpers
# ----------------------------
def _b64_to_pil_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def _b64_to_mask_array(b64: str) -> np.ndarray:
    """
    Frontend sends a PNG where painted region is white and background is transparent.
    We'll convert it to a boolean/binary mask.
    """
    raw = base64.b64decode(b64)
    m = Image.open(io.BytesIO(raw))

    # If RGBA, use alpha or luminance; if L/RGB, use luminance
    if m.mode == "RGBA":
        # Use alpha as mask if present; fallback to luminance.
        alpha = np.array(m.split()[-1], dtype=np.uint8)
        mask = (alpha > 0).astype(np.uint8)
    else:
        ml = m.convert("L")
        arr = np.array(ml, dtype=np.uint8)
        mask = (arr > 0).astype(np.uint8)

    return mask


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ----------------------------
# Model singleton (warm start)
# ----------------------------
TAG = os.environ.get("SAM3D_TAG", "hf")
CONFIG_PATH = os.environ.get("SAM3D_CONFIG", f"checkpoints/{TAG}/pipeline.yaml")

app = FastAPI(title="SAM3D Objects API")

_infer: Optional[Inference] = None


@app.on_event("startup")
def _startup():
    global _infer
    # Warm-load once per worker
    _infer = Inference(CONFIG_PATH, compile=False)


@app.get("/health")
def health():
    return {"ok": True, "config": CONFIG_PATH}


@app.post("/run", response_model=Sam3DResponse)
def run(req: Sam3DRequest):
    global _infer
    if _infer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_pil = _b64_to_pil_image(req.imageBase64)
        mask = _b64_to_mask_array(req.maskBase64)
        
        # Convert PIL -> numpy RGB (H, W, 3)
        image = np.array(image_pil, dtype=np.uint8)
        
        # (strongly recommended) resize mask to match image
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize((image.shape[1], image.shape[0]), resample=Image.NEAREST)
        mask = (np.array(mask_img) > 0).astype(np.uint8)
        
        out = _infer(image, mask, seed=req.options.seed)


        # --- Export requested formats ---
        glb_b64 = None
        ply_b64 = None

        with tempfile.TemporaryDirectory() as td:
            # 1) PLY: official readme shows gaussian splat export:
            #   output["gs"].save_ply("splat.ply")
            # :contentReference[oaicite:3]{index=3}
            if "ply" in req.options.output:
                ply_path = os.path.join(td, "out.ply")
                if "gs" in out and hasattr(out["gs"], "save_ply"):
                    out["gs"].save_ply(ply_path)
                    ply_b64 = _file_to_b64(ply_path)
                else:
                    raise RuntimeError("PLY requested but out['gs'].save_ply not available")

            # 2) GLB: depending on repo version, mesh export can differ.
            # We'll try a few common patterns and fail loudly if none exist.
            if "glb" in req.options.output:
                glb_path = os.path.join(td, "out.glb")

                # Try: out["mesh"].to_glb(path) or out["mesh"].save_glb(path) etc.
                mesh_obj = out.get("mesh") if isinstance(out, dict) else None
                if mesh_obj is not None:
                    if hasattr(mesh_obj, "to_glb"):
                        mesh_obj.to_glb(glb_path)
                    elif hasattr(mesh_obj, "save_glb"):
                        mesh_obj.save_glb(glb_path)
                    elif hasattr(mesh_obj, "export"):
                        mesh_obj.export(glb_path)
                    else:
                        raise RuntimeError("GLB requested but mesh export method not found on out['mesh']")
                    glb_b64 = _file_to_b64(glb_path)
                else:
                    raise RuntimeError("GLB requested but out['mesh'] is missing")

        return Sam3DResponse(
            glbBase64=glb_b64,
            plyBase64=ply_b64,
            meta={"seed": req.options.seed},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
