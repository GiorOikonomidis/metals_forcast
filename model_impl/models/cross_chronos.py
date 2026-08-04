"""
Model definition: cross-attention over index / news / covariate streams on top of
a frozen Chronos encoder. Architecture only — training lives in scripts/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from model_impl.models.blocks import CrossBlock
from model_impl.models.heads import build_head

if TYPE_CHECKING:
    from model_impl.utils.schemas import ModelConfig


class MultiCrossChronos(nn.Module):
    """
    Cross-attention EUR/USD ↔ News ↔ Covariates on top of a frozen Chronos encoder.
    """
    def __init__(self, vocab: int, n_covariates: int, model_cfg: ModelConfig,
                 ctx_len: int, pred_len: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.pred_len = pred_len
        cc = model_cfg.cross_chronos

        # Frozen Chronos encoder for the price stream
        self.enc_eur = AutoModelForSeq2SeqLM.from_pretrained(model_cfg.comp_enc).encoder
        for p in self.enc_eur.parameters():
            p.requires_grad = False
        self.enc_eur.eval()  # freezing stops gradients, not dropout — keep it eval

        # Text-like encoders for news/covariates (project to d_model then TransformerEncoder)
        self.news_proj = nn.Linear(cc.emb_dim_news, cc.d_model)

        self.no_covariates = n_covariates == 0
        if self.no_covariates:
            self.covariate_emb = nn.Parameter(torch.zeros(ctx_len, cc.d_model))
        else:
            self.covariate_proj = nn.Linear(n_covariates, cc.d_model)

        enc_layer = nn.TransformerEncoderLayer(cc.d_model, cc.n_heads, cc.d_ff,
                                               cc.dropout, batch_first=True)
        self.enc_news = nn.TransformerEncoder(enc_layer, cc.n_layers_txt)
        self.enc_covariate = nn.TransformerEncoder(enc_layer, cc.n_layers_txt)

        # Cross attention pairs
        self.eur_news_q  = CrossBlock(cc.d_model, cc.n_heads, cc.dropout)
        self.news_eur_q  = CrossBlock(cc.d_model, cc.n_heads, cc.dropout)
        self.eur_covariate_q  = CrossBlock(cc.d_model, cc.n_heads, cc.dropout)
        self.covariate_eur_q  = CrossBlock(cc.d_model, cc.n_heads, cc.dropout)
        self.news_covariate_q = CrossBlock(cc.d_model, cc.n_heads, cc.dropout)
        self.covariate_news_q = CrossBlock(cc.d_model, cc.n_heads, cc.dropout)

        # Normalization
        self.ln_eur  = nn.LayerNorm(cc.d_model)
        self.ln_news = nn.LayerNorm(cc.d_model)
        self.ln_covariate = nn.LayerNorm(cc.d_model)

        # Output head — logits over vocab per horizon step; swappable via cc.head
        self.head = build_head(cc.head, cc.d_model, pred_len, vocab)

    def train(self, mode: bool = True):
        """
            Keep the frozen Chronos encoder in eval mode always, even when the
            trainer flips the model to train() each epoch — its internal dropout
            must never perturb the frozen embeddings.
        """
        super().train(mode)
        self.enc_eur.eval()
        return self

    def mc_dropout(self, enable: bool) -> None:
        """
            Toggle dropout on the trainable submodules only, for MC-Dropout at
            inference. The frozen encoder is deliberately excluded so it stays
            deterministic. Call mc_dropout(True) before an MC loop and
            mc_dropout(False) after to restore full eval behaviour.
        """
        targets = [self.news_proj, self.enc_news, self.enc_covariate,
                   self.eur_news_q, self.news_eur_q, self.eur_covariate_q,
                   self.covariate_eur_q, self.news_covariate_q, self.covariate_news_q,
                   *self.head.mc_targets()]
        if not self.no_covariates:
            targets.append(self.covariate_proj)
        for mod in targets:
            mod.train(enable)

    def forward(self, tok_eur: torch.Tensor, seq_news: torch.Tensor,
                seq_covariate: torch.Tensor, mc: bool = False) -> torch.Tensor:
        h_eur  = self.enc_eur(input_ids=tok_eur).last_hidden_state
        h_news = self.enc_news(self.news_proj(seq_news))
        if self.no_covariates:
            h_covariate = self.enc_covariate(self.covariate_emb.unsqueeze(0).expand(seq_covariate.size(0), -1, -1))
        else:
            h_covariate = self.enc_covariate(self.covariate_proj(seq_covariate))

        # One-step bidirectional cross-attention
        h_eur  = self.eur_news_q(h_eur,  h_news)
        h_news = self.news_eur_q(h_news, h_eur)
        h_eur  = self.eur_covariate_q(h_eur,  h_covariate)
        h_covariate = self.covariate_eur_q(h_covariate, h_eur)
        h_news = self.news_covariate_q(h_news, h_covariate)
        h_covariate = self.covariate_news_q(h_covariate, h_news)

        h_eur  = self.ln_eur(h_eur)
        h_news = self.ln_news(h_news)
        h_covariate = self.ln_covariate(h_covariate)

        # Fuse the full per-timestep sequences from the three streams — heads
        # that only need the last step (Linear, MLP) slice it themselves;
        # sequence-aware heads (LSTM, Transformer) see the whole context.
        fused_seq = torch.cat([h_eur, h_news, h_covariate], dim=-1)

        # MC-Dropout is controlled by the caller via model.mc_dropout(True/False)
        # around the sampling loop — not here, so the frozen encoder is never
        # flipped to train mode and the model is not left in train mode after eval.
        logits = self.head(fused_seq, tok_eur)
        return logits
