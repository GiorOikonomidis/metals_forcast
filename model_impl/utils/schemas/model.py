"""Model configuration: the frozen encoder and the cross-attention architecture."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CrossChronosConfig:
    emb_dim_news: int = 768
    d_model: int = 768
    n_heads: int = 8
    n_layers_txt: int = 3
    d_ff: int = 1024
    dropout: float = 0.2
    head: str = "linear"  # linear | mlp | lstm | transformer


@dataclass(frozen=True)
class ModelConfig:
    comp_enc: str = "amazon/chronos-t5-base"
    cross_chronos: CrossChronosConfig = field(default_factory=CrossChronosConfig)
