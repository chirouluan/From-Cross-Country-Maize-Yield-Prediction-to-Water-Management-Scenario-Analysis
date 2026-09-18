"""Shared configuration for the three research stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cybench.config import REPO_DIR, PATH_OUTPUT_DIR

DEFAULT_TRANSFER_COUNTRIES = ("PT", "ZA", "ES", "IT", "IN", "MX", "ZM")
DEFAULT_STAGE1_COUNTRIES = ("CN",) + DEFAULT_TRANSFER_COUNTRIES
DEFAULT_REGRESSORS = ("ridge", "random_forest", "xgboost")
PROJECT_ROOT = Path(REPO_DIR)
DEFAULT_QWEN_BASE_DIR = PROJECT_ROOT / "LLM" / "Qwen3.5-9B"
DEFAULT_TABPFN_BASE_CHECKPOINT = (
    PROJECT_ROOT / "tabpfn_3" / "TabPFN" / "tabpfn-v3-regressor-v3_default.ckpt"
)
DEFAULT_TABPFN_FINETUNED_CHECKPOINT = (
    PROJECT_ROOT
    / "tabpfn_3"
    / "Agronomy TabPFN"
    / "tabpfn-v3-regressor-cn_finetuned_full.ckpt"
)
DEFAULT_STAGE1_OUTPUT_DIR = (
    Path(PATH_OUTPUT_DIR) / "stages" / "stage1_feature_engineering"
)
DEFAULT_CLIMATE_RAG_METRICS = (
    PROJECT_ROOT
    / "Climate_rag_knowledge_base"
    / "processed"
    / "climate_rag_metrics.csv"
)
DEFAULT_STAGE2_OUTPUT_DIR = (
    Path(PATH_OUTPUT_DIR) / "stages" / "stage2_tabpfn_transfer" / "evaluation"
)


@dataclass(frozen=True)
class LLMVariant:
    name: str
    model_dir: Path


def qwen_base_variant(
    base_dir: str | os.PathLike | None = None,
) -> LLMVariant:
    """Return the canonical Base Qwen used by both plain and RAG branches."""
    base = Path(base_dir or os.getenv("CYBENCH_QWEN_BASE", DEFAULT_QWEN_BASE_DIR))
    return LLMVariant("qwen_base", base)


def validate_llm_variant(variant: LLMVariant) -> Path:
    """Require a loadable full Qwen model directory, not an adapter-only folder."""
    directory = variant.model_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{variant.name} model directory not found: {directory}"
        )
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(f"{variant.name} config.json not found: {directory}")
    has_weights = (directory / "model.safetensors").is_file() or (
        directory / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        raise FileNotFoundError(f"{variant.name} model weights not found: {directory}")
    return directory


def tabpfn_checkpoints(
    base: str | os.PathLike | None = None,
    finetuned: str | os.PathLike | None = None,
) -> tuple[Path, Path]:
    """Return configured original and agronomy-finetuned TabPFN checkpoints."""
    base_path = Path(
        base or os.getenv("CYBENCH_TABPFN_BASE", DEFAULT_TABPFN_BASE_CHECKPOINT)
    )
    tuned_path = Path(
        finetuned
        or os.getenv("CYBENCH_TABPFN_FINETUNED", DEFAULT_TABPFN_FINETUNED_CHECKPOINT)
    )
    return (
        require_file(base_path, "Base TabPFN checkpoint"),
        require_file(tuned_path, "Agronomy-finetuned TabPFN checkpoint"),
    )


def stage_output_dir(stage: str, output_dir: str | None = None) -> Path:
    path = Path(output_dir) if output_dir else Path(PATH_OUTPUT_DIR) / "stages" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: str | os.PathLike, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
