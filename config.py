import torch
from types import SimpleNamespace

BACKEND = "ollama"  # "openmed" | "ollama" | "hf"


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


AUTO_DEVICE = _detect_device()

whisper_cfg = SimpleNamespace(
    model_size="base",   # tiny | base | small | medium | large-v3-turbo
    language="auto",     # "id" | "en" | "auto"
)

ollama_cfg = SimpleNamespace(
    model="llama3.2",
)

openmed_cfg = SimpleNamespace(
    deidentify_before_soap=True,
    ner_model="openmed/medical-ner",
    device=AUTO_DEVICE,
    deidentify_method="replace",
)

output_cfg = SimpleNamespace(
    output_dir="outputs",
)
