"""Head registry: pick a head architecture by name (see utils/schemas/model.py CrossChronosConfig.head)."""

from __future__ import annotations

from model_impl.models.heads.linear_head import LinearHead
from model_impl.models.heads.lstm_head import LSTMHead
from model_impl.models.heads.mlp_head import MLPHead
from model_impl.models.heads.transformer_head import TransformerHead

_REGISTRY = {
    "linear": LinearHead,
    "mlp": MLPHead,
    "lstm": LSTMHead,
    "transformer": TransformerHead,
}


def build_head(name: str, d_model: int, pred_len: int, vocab: int):
    """
    Parameters
    ----------
    name : str
        One of "linear", "mlp", "lstm", "transformer".
    d_model : int
        Per-stream hidden size (fused width is d_model*3).
    pred_len : int
        Number of horizon days to predict.
    vocab : int
        Chronos token-bin count.

    Returns
    -------
    nn.Module
        A head instance implementing forward(fused_seq, tok_eur, y) -> (B,PRED,vocab)
        and mc_targets() -> list[nn.Module].
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown head {name!r}; expected one of {sorted(_REGISTRY)}") from None
    return cls(d_model, pred_len, vocab)
