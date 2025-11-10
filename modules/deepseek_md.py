# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import Dict, Any
import torch
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_MODEL_ID = "Jalea96/DeepSeek-OCR-bnb-4bit-NF4"
_tokenizer = None
_model = None

def _ensure_model():
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        _MODEL_ID,
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    return _tokenizer, _model

def image_to_markdown(image_path: str, out_dir: str) -> Dict[str, Any]:
    tok, model = _ensure_model()
    os.makedirs(out_dir, exist_ok=True)
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    res = model.infer(
        tok,
        prompt=prompt,
        image_file=image_path,
        output_path=out_dir,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True,
    )
    md = res.get("markdown") if isinstance(res, dict) else ""
    txt = res.get("text") if isinstance(res, dict) else str(res)
    if not md:
        md = txt or ""
    return {"markdown": md, "text": txt, "raw": res}
