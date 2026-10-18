"""FastAPI inference service for DistilBERT with LoRA sentiment classification."""

import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.model import load_adapter, build_model
from src.lora import merge_lora_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve")

# Global state for loaded model and tokenizer
state: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": "cpu",
    "adapter_loaded": False,
    "base_model_name": "distilbert-base-uncased",
}

LABEL_MAP = {
    0: "negative",
    1: "positive",
}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input sentence for sentiment classification", examples=["This movie was wonderful and deeply moving!"])


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment label ('positive' or 'negative')")
    score: float = Field(..., description="Confidence probability for predicted class (0.0 to 1.0)")
    class_id: int = Field(..., description="Integer class ID (0 for negative, 1 for positive)")


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter_loaded: bool
    device: str


def initialize_model_and_tokenizer(adapters_dir: str = "adapters", base_model_name: str = "distilbert-base-uncased") -> None:
    """Loads tokenizer and model with LoRA adapter if present, merging weights for fast inference."""
    device = torch.device("cpu")
    state["device"] = str(device)
    state["base_model_name"] = base_model_name

    logger.info("Loading tokenizer for %s...", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    state["tokenizer"] = tokenizer

    adapter_config_path = os.path.join(adapters_dir, "adapter_config.json")
    adapter_weights_path = os.path.join(adapters_dir, "adapter_model.pt")

    if os.path.exists(adapter_config_path) and os.path.exists(adapter_weights_path):
        logger.info("Found trained LoRA adapter in '%s'. Loading...", adapters_dir)
        try:
            model, config = load_adapter(adapters_dir, device=device)
            # Merge LoRA weights into base weights for zero-overhead inference
            merge_lora_model(model)
            model.eval()
            state["model"] = model
            state["adapter_loaded"] = True
            state["base_model_name"] = config.get("base_model_name", base_model_name)
            logger.info("Successfully loaded and merged LoRA adapter.")
            return
        except Exception as e:
            logger.warning("Could not load adapter from %s: %s. Falling back to base model.", adapters_dir, e)

    # Fallback to base model if adapter is not yet available
    logger.info("Loading base classification model %s without adapter...", base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    model.eval()
    model.to(device)
    state["model"] = model
    state["adapter_loaded"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_model_and_tokenizer()
    yield
    # Shutdown
    state.clear()


app = FastAPI(
    title="DistilBERT LoRA Sentiment Inference API",
    description="High-performance inference API serving DistilBERT fine-tuned with scratch LoRA on SST-2.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Health check endpoint confirming service status and loaded model details."""
    is_healthy = state["model"] is not None and state["tokenizer"] is not None
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model=state["base_model_name"],
        adapter_loaded=state["adapter_loaded"],
        device=state["device"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Classifies sentiment of input text returning label and confidence score."""
    model = state.get("model")
    tokenizer = state.get("tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet initialized or loaded.",
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text payload cannot be empty or purely whitespace.",
        )

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = F.softmax(logits, dim=-1)[0]
            pred_class = torch.argmax(probs).item()
            pred_score = float(probs[pred_class].item())

        return PredictResponse(
            label=LABEL_MAP.get(pred_class, str(pred_class)),
            score=round(pred_score, 5),
            class_id=pred_class,
        )
    except Exception as e:
        logger.error("Inference failure on text: %s. Error: %s", text, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=False)
