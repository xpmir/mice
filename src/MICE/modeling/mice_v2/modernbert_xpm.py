"""
Experimaestro-IR integration for ModernBERT MICE V2 architecture.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict
import logging
import shutil
import json
import os
import math
import atexit

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification

from experimaestro import LightweightTask, Param, Meta, Constant, field
from datamaestro_ir.data.base import IDTextRecord
from xpmir.text.tokenizers import TokenizerOptions, TokenizedTexts
from xpmir.rankers.scorer import AbstractModuleScorer
from xpmir.letor.records import BaseItems
from xpmir.text.encoders import (
    TextEncoderBase,
    EncoderOutput,
    TokensEncoderOutput,
    TokensRepresentationOutput,
)
from xpm_torch.utils import to_device
from xpm_torch.trainers import TrainingHook, TrainerContext
from xpm_torch.losses import Loss
from xpm_torch.metrics import ScalarMetric
from xpm_torch.module import (
    ModuleLoader,
    SimpleModuleLoader,
    ReadmeSection,
)
from xpm_torch.utils.fabric import fallback_fa2_if_incompatible_precision
from MICE.modeling.mice import MICEQueryDocTokenizer, MICETokenizedTexts, QueryDocInput
from MICE.modeling.mice_v2.modernbert_hf import (
    ModernBertMiceConfig,
    ModernBertMiceForSequenceClassification,
)
from MICE.modeling.mice_v2.compress_init import initialize_compressed_top_layers


logger = logging.getLogger(__name__)


def compute_model_norm_metrics(
    hf_model: nn.Module,
) -> dict:
    """Compute Frobenius norms of W_V and W_O weights."""
    metrics = {}
    if hasattr(hf_model, "get_wv_matrices"):
        wv_list = hf_model.get_wv_matrices()
        if wv_list:
            norms = [torch.norm(w, p="fro") for w in wv_list]
            metrics["wv_norm/mean"] = torch.stack(norms).mean()
            metrics["wv_norm/layer_0"] = norms[0]

    if hasattr(hf_model, "get_wo_matrices"):
        wo_list = hf_model.get_wo_matrices()
        if wo_list:
            norms = [torch.norm(w, p="fro") for w in wo_list]
            metrics["wo_norm/mean"] = torch.stack(norms).mean()
            metrics["wo_norm/layer_0"] = norms[0]

    return metrics


def compute_token_split_metrics(
    valid_utilities: torch.Tensor,
    sparsity_threshold: float = 0.5,
    valid_doc_hidden: Optional[torch.Tensor] = None,
) -> dict:
    """Compute token-level metrics split by sparsity_threshold (kept vs suppressed)."""
    metrics = {}
    utils = valid_utilities.view(-1)
    kept_mask = utils >= sparsity_threshold
    suppressed_mask = utils < sparsity_threshold

    if kept_mask.any():
        metrics["utility/mean_kept"] = utils[kept_mask].mean()
    if suppressed_mask.any():
        metrics["utility/mean_suppressed"] = utils[suppressed_mask].mean()

    if valid_doc_hidden is not None:
        doc_h = valid_doc_hidden.view(-1, valid_doc_hidden.shape[-1])
        if doc_h.shape[0] == utils.shape[0]:
            per_token_doc_norm = torch.norm(doc_h, p=2, dim=-1)
            per_token_gated_norm = torch.norm(utils.unsqueeze(-1) * doc_h, p=2, dim=-1)

            metrics["doc_hidden_norm/mean"] = per_token_doc_norm.mean()
            metrics["gated_doc_norm/mean"] = per_token_gated_norm.mean()

            if kept_mask.any():
                metrics["doc_hidden_norm/mean_kept"] = per_token_doc_norm[
                    kept_mask
                ].mean()
                metrics["gated_doc_norm/mean_kept"] = per_token_gated_norm[
                    kept_mask
                ].mean()

            if suppressed_mask.any():
                metrics["doc_hidden_norm/mean_suppressed"] = per_token_doc_norm[
                    suppressed_mask
                ].mean()
                metrics["gated_doc_norm/mean_suppressed"] = per_token_gated_norm[
                    suppressed_mask
                ].mean()

            if kept_mask.any() and suppressed_mask.any():
                gated_kept = metrics["gated_doc_norm/mean_kept"]
                gated_suppressed = metrics["gated_doc_norm/mean_suppressed"]
                metrics["gated_contrast_ratio"] = gated_kept / (gated_suppressed + 1e-6)

    return metrics


def compute_linear_schedule(
    start_val: float,
    end_val: Optional[float],
    step: int,
    total_steps: int,
    min_val: float = 1e-4,
) -> float:
    """Linearly interpolate from start_val to end_val over total_steps.
    If end_val is None, returns start_val.
    """
    if end_val is None:
        return start_val
    if total_steps <= 0:
        return end_val
    progress = min(1.0, max(0.0, step / total_steps))
    val = start_val - (start_val - end_val) * progress
    return max(val, min_val)


def compute_cosine_schedule(
    start_val: float,
    end_val: Optional[float],
    step: int,
    total_steps: int,
    min_val: float = 1e-4,
) -> float:
    """Cosine interpolate from start_val to end_val over total_steps.
    If end_val is None, returns start_val.
    """
    if end_val is None:
        return start_val
    if total_steps <= 0:
        return end_val
    progress = min(1.0, max(0.0, step / total_steps))
    val = end_val + 0.5 * (start_val - end_val) * (1.0 + math.cos(math.pi * progress))
    return max(val, min_val)


def compute_exponential_schedule(
    start_val: float,
    end_val: Optional[float],
    step: int,
    total_steps: int,
    min_val: float = 1e-4,
) -> float:
    """Exponentially interpolate from start_val to end_val over total_steps.
    If end_val is None, returns start_val.
    """
    if end_val is None:
        return start_val
    if total_steps <= 0:
        return end_val
    progress = min(1.0, max(0.0, step / total_steps))
    s = max(start_val, min_val)
    e = max(end_val, min_val)
    val = s * ((e / s) ** progress)
    return max(val, min_val)


def compute_temperature_schedule(
    start_val: float,
    end_val: Optional[float],
    step: int,
    total_steps: int,
    schedule: str = "linear",
    min_val: float = 1e-4,
) -> float:
    """Computes scheduled temperature using the specified schedule ('linear', 'cosine', or 'exp')."""
    if schedule == "linear":
        return compute_linear_schedule(start_val, end_val, step, total_steps, min_val)
    elif schedule == "cosine":
        return compute_cosine_schedule(start_val, end_val, step, total_steps, min_val)
    elif schedule in ("exp", "exponential"):
        return compute_exponential_schedule(
            start_val, end_val, step, total_steps, min_val
        )
    else:
        raise ValueError(
            f"Unknown temperature schedule '{schedule}'. Supported: 'linear', 'cosine', 'exp'."
        )


def update_utility_temperature(
    model: Optional[nn.Module],
    info: TrainerContext,
    temp_start: float,
    temp_end: Optional[float],
    temp_anneal_steps: Optional[int] = None,
    schedule: str = "linear",
    nrecords: int = 1,
):
    """Updates model.utility_predictor.temperature if temp_end is set, and logs utility/temperature metric."""
    target_model = model
    if hasattr(target_model, "hf_model"):
        target_model = target_model.hf_model

    if target_model is None or getattr(target_model, "utility_predictor", None) is None:
        return

    if temp_end is not None:
        total_steps = (
            temp_anneal_steps
            if temp_anneal_steps is not None
            else (info.max_epoch * info.steps_per_epoch)
        )
        cur_temp = compute_temperature_schedule(
            temp_start, temp_end, info.state.step, total_steps, schedule=schedule
        )
        target_model.utility_predictor.temperature.fill_(cur_temp)

    if getattr(info, "metrics", None) is not None:
        info.metrics.add(
            ScalarMetric(
                "utility/temperature",
                float(target_model.utility_predictor.temperature.item()),
                nrecords,
            )
        )


class UtilityRegularizer(TrainingHook):
    """Training hook for computing token utility L1 regularization loss and sparsity metrics"""

    lambda_reg: Param[float] = field(default=0.1)
    """Regularization weight for utility predictor L1 loss"""

    budget: Param[Optional[float]] = field(default=None, ignore_default=True)
    """Optional target retention budget (e.g. 0.3 for 30% retention). If set, loss is relu(mean_utility - budget), halting regularization when mean_utility <= budget."""

    l2_reg: Param[bool] = field(default=False, ignore_default=True)
    """Whether to square the regularization loss (L2 penalty instead of L1). Provides smoother gradients especially when combined with budget."""

    temp_start: Param[float] = field(default=1.0, ignore_default=True)
    """Starting temperature for utility predictor sigmoid (default 1.0)"""

    temp_end: Param[Optional[float]] = field(default=None, ignore_default=True)
    """Ending temperature for linear annealing (e.g. 0.1). If None, temperature remains fixed at temp_start."""

    temp_anneal_steps: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Number of steps over which to anneal temperature. If None, anneals over total training steps (max_epoch * steps_per_epoch)."""

    temp_schedule: Param[str] = field(default="linear", ignore_default=True)
    """Temperature annealing schedule: 'linear' (default), 'cosine', or 'exp'."""

    sparsity_threshold: Meta[float] = field(default=0.5)
    """Threshold below which a token utility score is considered suppressed / sparse (default 0.5) affects sparsity metric computation only"""

    histogram_interval: Meta[int] = field(default=0)
    """Interval in steps for writing utility distribution histograms to TensorBoard (0 to disable)"""

    version: Constant[str] = "1.0.0"

    def __call__(
        self,
        info: TrainerContext,
        utilities: torch.Tensor,
        doc_mask: Optional[torch.Tensor] = None,
        p_active: Optional[torch.Tensor] = None,
        doc_hidden: Optional[torch.Tensor] = None,
        hf_model: Optional[nn.Module] = None,
    ):
        if self.lambda_reg <= 0.0 or utilities is None:
            return

        if (
            doc_mask is not None
            and utilities.ndim == 3
            and utilities.shape[0] == doc_mask.shape[0]
            and utilities.shape[1] == doc_mask.shape[1]
        ):  # we are using sdpa - need to mask tokens
            mask = doc_mask.unsqueeze(-1).bool()
            valid_utilities = utilities.masked_select(mask)
            if valid_utilities.numel() > 0:
                mean_utility = valid_utilities.mean()
                sparsity = (valid_utilities < self.sparsity_threshold).float().mean()
            else:
                mean_utility = utilities.mean()
                sparsity = (utilities < self.sparsity_threshold).float().mean()
        else:
            # under FA2, padding tokens are already removed, so we can just take the mean over all tokens
            mean_utility = utilities.mean()
            sparsity = (utilities < self.sparsity_threshold).float().mean()
            valid_utilities = utilities.view(-1)

        target_model = hf_model or getattr(info, "model", None)
        model_to_inspect = getattr(target_model, "hf_model", target_model)
        if p_active is None:
            p_active = getattr(model_to_inspect, "last_p_active", None)

        if p_active is not None:
            # Hard Concrete L0 loss formulation over exact retention probability
            if (
                doc_mask is not None
                and p_active.ndim == 3
                and p_active.shape[0] == doc_mask.shape[0]
                and p_active.shape[1] == doc_mask.shape[1]
            ):
                mask = doc_mask.unsqueeze(-1).bool()
                valid_p_active = p_active.masked_select(mask)
                mean_p_active = (
                    valid_p_active.mean()
                    if valid_p_active.numel() > 0
                    else p_active.mean()
                )
            else:
                mean_p_active = p_active.mean()

            if self.budget is not None:
                diff = F.relu(mean_p_active - self.budget)
            else:
                diff = mean_p_active

            reg_loss = diff**2 if self.l2_reg else diff
        else:
            if self.budget is not None:
                diff = F.relu(mean_utility - self.budget)
            else:
                diff = mean_utility

            reg_loss = diff**2 if self.l2_reg else diff

        info.add_loss(Loss("utility_reg", reg_loss, self.lambda_reg))

        nrecords = doc_mask.shape[0] if doc_mask is not None else 1

        update_utility_temperature(
            target_model,
            info,
            self.temp_start,
            self.temp_end,
            self.temp_anneal_steps,
            schedule=self.temp_schedule,
            nrecords=nrecords,
        )

        if getattr(info, "metrics", None) is not None:
            info.metrics.add(
                ScalarMetric(
                    "utility_reg",
                    float((self.lambda_reg * reg_loss).item()),
                    nrecords,
                )
            )
            info.metrics.add(
                ScalarMetric("utility/mean", float(mean_utility.item()), nrecords)
            )
            info.metrics.add(
                ScalarMetric("utility/sparsity", float(sparsity.item()), nrecords)
            )
            if self.budget is not None:
                info.metrics.add(
                    ScalarMetric("utility/budget", float(self.budget), nrecords)
                )
                info.metrics.add(
                    ScalarMetric("utility/over_budget", float(diff.item()), nrecords)
                )

            if p_active is not None:
                info.metrics.add(
                    ScalarMetric(
                        "l0/p_active_mean", float(mean_p_active.item()), nrecords
                    )
                )
                info.metrics.add(
                    ScalarMetric(
                        "l0/exact_zeros",
                        float((valid_utilities == 0.0).float().mean().item()),
                        nrecords,
                    )
                )
                info.metrics.add(
                    ScalarMetric(
                        "l0/exact_ones",
                        float((valid_utilities == 1.0).float().mean().item()),
                        nrecords,
                    )
                )
                info.metrics.add(
                    ScalarMetric(
                        "l0/intermediate",
                        float(
                            ((valid_utilities > 0.0) & (valid_utilities < 1.0))
                            .float()
                            .mean()
                            .item()
                        ),
                        nrecords,
                    )
                )

            # Monitor Model Matrix Norms (W_V, W_O, Classifier Head)
            if target_model is not None:
                if hasattr(target_model, "hf_model"):
                    target_model = target_model.hf_model
                norm_metrics = compute_model_norm_metrics(target_model)
                for k, v in norm_metrics.items():
                    info.metrics.add(ScalarMetric(k, float(v.item()), nrecords))

            # Monitor Token-Level Diagnostics (Kept vs Suppressed)
            doc_h = (
                doc_hidden
                if doc_hidden is not None
                else (
                    getattr(target_model, "last_doc_hidden", None)
                    if target_model is not None
                    else None
                )
            )
            split_metrics = compute_token_split_metrics(
                valid_utilities,
                sparsity_threshold=self.sparsity_threshold,
                valid_doc_hidden=doc_h,
            )
            for k, v in split_metrics.items():
                info.metrics.add(ScalarMetric(k, float(v.item()), nrecords))

        # Log histogram at interval boundaries
        metrics_logger = getattr(info, "metrics_logger", None)
        if (
            self.histogram_interval > 0
            and metrics_logger is not None
            and (info.state.step % self.histogram_interval == 0)
            and valid_utilities.numel() > 0
        ):
            metrics_logger.log_histogram(
                "train/utility/distribution",
                valid_utilities.detach().float().cpu().numpy(),
                step=info.state.step,
            )


class GatedValueRegularizer(TrainingHook):
    """Training hook for regularizing the norm of gated document value representations (u * W_V * h)"""

    lambda_reg: Param[float] = field(default=0.1)
    """Regularization weight for gated value norm loss"""

    budget: Param[Optional[float]] = field(default=None, ignore_default=True)
    """Optional target budget for gated value norm. If set, loss is relu(mean_v_norm - budget)."""

    l2_reg: Param[bool] = field(default=False, ignore_default=True)
    """Whether to square the regularization loss. Provides smoother gradients especially when combined with budget."""

    temp_start: Param[float] = field(default=1.0, ignore_default=True)
    """Starting temperature for utility predictor sigmoid (default 1.0)"""

    temp_end: Param[Optional[float]] = field(default=None, ignore_default=True)
    """Ending temperature for linear annealing (e.g. 0.1). If None, temperature remains fixed at temp_start."""

    temp_anneal_steps: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Number of steps over which to anneal temperature. If None, anneals over total training steps (max_epoch * steps_per_epoch)."""

    temp_schedule: Param[str] = field(default="linear", ignore_default=True)
    """Temperature annealing schedule: 'linear' (default), 'cosine', or 'exp'."""

    sparsity_threshold: Meta[float] = field(default=0.5)
    """Threshold below which a token utility score is considered suppressed / sparse"""

    histogram_interval: Meta[int] = field(default=0)
    """Interval in steps for writing utility distribution histograms to TensorBoard (0 to disable)"""

    version: Constant[str] = "1.0.0"

    def __call__(
        self,
        info: TrainerContext,
        model: nn.Module,
        utilities: torch.Tensor,
        doc_hidden: torch.Tensor,
        doc_mask: Optional[torch.Tensor] = None,
    ):
        if self.lambda_reg <= 0.0 or utilities is None or doc_hidden is None:
            return

        target_model = model
        if hasattr(target_model, "hf_model"):
            target_model = target_model.hf_model

        wv_list = (
            target_model.get_wv_matrices()
            if hasattr(target_model, "get_wv_matrices")
            else []
        )
        if not wv_list:
            logger.warning(
                "GatedValueRegularizer: No W_V matrices found in target model."
            )
            return

        # Use W_V of the first top interaction layer
        W_V = wv_list[0]
        top_layer0_attn = target_model.top_layers[0].cross_attn
        hidden_size = top_layer0_attn.config.hidden_size
        bias_V = (
            top_layer0_attn.Wqkv.bias[2 * hidden_size : 3 * hidden_size]
            if top_layer0_attn.Wqkv.bias is not None
            else None
        )

        # Compute value projection V(h) = h @ W_V^T + bias_V
        value_states = F.linear(doc_hidden, W_V, bias_V)  # [B, L, D] or [1, Total_D, D]
        # Compute gated value representations: u * V(h)
        gated_value_states = utilities * value_states  # broadcast u over D

        # Compute per-token L2 norm normalized by sqrt(D)
        per_token_v_norm = torch.norm(gated_value_states, p=2, dim=-1) / (
            hidden_size**0.5
        )

        if (
            doc_mask is not None
            and utilities.ndim == 3
            and utilities.shape[0] == doc_mask.shape[0]
            and utilities.shape[1] == doc_mask.shape[1]
        ):
            # Padded 3D SDPA path
            mask = doc_mask.bool()
            valid_v_norms = per_token_v_norm.masked_select(mask)
            valid_utilities = utilities.masked_select(mask.unsqueeze(-1))
            valid_doc_h = (
                doc_hidden.masked_select(mask.unsqueeze(-1)).view(-1, hidden_size)
                if doc_hidden is not None
                else None
            )
            if valid_v_norms.numel() > 0:
                mean_v_norm = valid_v_norms.mean()
                mean_utility = valid_utilities.mean()
                sparsity = (valid_utilities < self.sparsity_threshold).float().mean()
            else:
                mean_v_norm = per_token_v_norm.mean()
                mean_utility = utilities.mean()
                sparsity = (utilities < self.sparsity_threshold).float().mean()
        else:
            # Unpadded 1D FA2 path
            mean_v_norm = per_token_v_norm.mean()
            mean_utility = utilities.mean()
            sparsity = (utilities < self.sparsity_threshold).float().mean()
            valid_v_norms = per_token_v_norm.view(-1)
            valid_utilities = utilities.view(-1)
            valid_doc_h = (
                doc_hidden.view(-1, hidden_size) if doc_hidden is not None else None
            )

        if self.budget is not None:
            diff = F.relu(mean_v_norm - self.budget)
        else:
            diff = mean_v_norm

        reg_loss = diff**2 if self.l2_reg else diff

        info.add_loss(Loss("gated_value_reg", reg_loss, self.lambda_reg))

        nrecords = doc_mask.shape[0] if doc_mask is not None else 1
        update_utility_temperature(
            target_model,
            info,
            self.temp_start,
            self.temp_end,
            self.temp_anneal_steps,
            schedule=self.temp_schedule,
            nrecords=nrecords,
        )

        if getattr(info, "metrics", None) is not None:
            info.metrics.add(
                ScalarMetric(
                    "gated_value_reg",
                    float((self.lambda_reg * reg_loss).item()),
                    nrecords,
                )
            )
            info.metrics.add(
                ScalarMetric("value_norm/mean", float(mean_v_norm.item()), nrecords)
            )
            info.metrics.add(
                ScalarMetric("utility/mean", float(mean_utility.item()), nrecords)
            )
            info.metrics.add(
                ScalarMetric("utility/sparsity", float(sparsity.item()), nrecords)
            )
            if self.budget is not None:
                info.metrics.add(
                    ScalarMetric("value_norm/budget", float(self.budget), nrecords)
                )
                info.metrics.add(
                    ScalarMetric("value_norm/over_budget", float(diff.item()), nrecords)
                )

            # Monitor Model Matrix Norms (W_V, W_O, Classifier Head)
            norm_metrics = compute_model_norm_metrics(target_model)
            for k, v in norm_metrics.items():
                info.metrics.add(ScalarMetric(k, float(v.item()), nrecords))

            # Monitor Token-Level Diagnostics (Kept vs Suppressed)
            split_metrics = compute_token_split_metrics(
                valid_utilities,
                sparsity_threshold=self.sparsity_threshold,
                valid_doc_hidden=valid_doc_h,
            )
            for k, v in split_metrics.items():
                info.metrics.add(ScalarMetric(k, float(v.item()), nrecords))

        # Log histograms
        metrics_logger = getattr(info, "metrics_logger", None)
        if (
            self.histogram_interval > 0
            and metrics_logger is not None
            and (info.state.step % self.histogram_interval == 0)
            and valid_v_norms.numel() > 0
        ):
            metrics_logger.log_histogram(
                "train/value_norm/distribution",
                valid_v_norms.detach().float().cpu().numpy(),
                step=info.state.step,
            )
            metrics_logger.log_histogram(
                "train/utility/distribution",
                valid_utilities.detach().float().cpu().numpy(),
                step=info.state.step,
            )


class InitMICEV2ModernBERTFromHFID(LightweightTask):
    """Task to initialize weights of ModernBertMiceForSequenceClassification from HF checkpoint"""

    scorer: Param["ModernBertMiceScorer"]
    freeze_bottom_layers: Param[bool] = field(default=False, ignore_default=True)
    """Whether to freeze the bottom contextualization layers and embeddings after initializing from backbone"""

    def execute(self):
        scorer = self.scorer
        logger.info(f"Building MICE from ModernBERT Encoder {scorer.hf_id}")
        scorer.initialize()
        full_backbone = AutoModelForSequenceClassification.from_pretrained(scorer.hf_id)

        # 1. Embeddings
        if hasattr(full_backbone.model, "embeddings"):
            logger.info("Seeding embeddings from backbone")
            scorer.hf_model.model.embeddings.load_state_dict(
                full_backbone.model.embeddings.state_dict()
            )

        # 2. Bottom contextualization layers
        if hasattr(full_backbone.model, "layers"):
            logger.info(
                f"Seeding {scorer.n_contextualization_layers} bottom layers from backbone"
            )
            for i in range(scorer.n_contextualization_layers):
                if i < len(full_backbone.model.layers):
                    scorer.hf_model.model.layers[i].load_state_dict(
                        full_backbone.model.layers[i].state_dict()
                    )

        # 3. Top interaction layers
        compress_dim = getattr(scorer.hf_model.config, "compress_dim", 1.0)
        # `mice_v2_scorer()` already resolves "seed"'s compress_dim-dependent fallback
        # (seed from the pretrained checkpoint at compress_dim == 1.0, else "random")
        # before building this scorer, so scorer.compress_init_option is always one of
        # "seed" (compress_dim == 1.0 only), "random", "random_projection", or
        # "svd_projection" by the time it gets here -- no further resolution needed.
        compress_init_option = scorer.compress_init_option

        if compress_init_option == "random":
            logger.info(
                "compress_init_option='random': leaving top interaction layers at "
                "their own default (unseeded) initialization."
            )
        elif compress_init_option == "seed":
            if hasattr(full_backbone.model, "layers"):
                logger.info(
                    f"Seeding {len(scorer.hf_model.top_layers)} top layers from backbone"
                )
                for idx, top_layer in enumerate(scorer.hf_model.top_layers):
                    src_layer_idx = scorer.n_contextualization_layers + idx
                    if src_layer_idx < len(full_backbone.model.layers):
                        src_layer = full_backbone.model.layers[src_layer_idx]
                        norm_to_copy = (
                            src_layer.attn_norm
                            if hasattr(src_layer, "attn_norm")
                            and isinstance(src_layer.attn_norm, nn.LayerNorm)
                            else (
                                src_layer.mlp_norm
                                if hasattr(src_layer, "mlp_norm")
                                and isinstance(src_layer.mlp_norm, nn.LayerNorm)
                                else None
                            )
                        )
                        # 1. Unconditionally seed self-attention and MLP modules
                        if norm_to_copy is not None:
                            top_layer.self_attn_norm.load_state_dict(
                                norm_to_copy.state_dict()
                            )
                        if hasattr(src_layer, "mlp_norm") and isinstance(
                            src_layer.mlp_norm, nn.LayerNorm
                        ):
                            top_layer.mlp_norm.load_state_dict(
                                src_layer.mlp_norm.state_dict()
                            )

                        top_layer.self_attn.load_state_dict(src_layer.attn.state_dict())
                        top_layer.mlp.load_state_dict(src_layer.mlp.state_dict())

                        # 2. Conditionally seed cross-attention modules if enabled
                        if top_layer.cross_attn is not None:
                            if norm_to_copy is not None:
                                top_layer.cross_attn_norm.load_state_dict(
                                    norm_to_copy.state_dict()
                                )
                                if isinstance(
                                    top_layer.doc_cross_attn_norm, nn.LayerNorm
                                ):
                                    top_layer.doc_cross_attn_norm.load_state_dict(
                                        norm_to_copy.state_dict()
                                    )

                            top_layer.cross_attn.Wqkv.load_state_dict(
                                src_layer.attn.Wqkv.state_dict()
                            )
                            top_layer.cross_attn.Wo.load_state_dict(
                                src_layer.attn.Wo.state_dict()
                            )
        else:
            logger.info(
                f"Projection-derived init (compress_dim={compress_dim}): "
                f"initializing {len(scorer.hf_model.top_layers)} top layers via "
                f"compress_init_option={compress_init_option!r}"
            )
            initialize_compressed_top_layers(
                scorer.hf_model,
                full_backbone,
                scorer.n_contextualization_layers,
                compress_init_option,
            )

        # 4. Rotary embeddings
        logger.info("Seeding rotary_emb from backbone")
        scorer.hf_model.model.rotary_emb.load_state_dict(
            full_backbone.model.rotary_emb.state_dict()
        )

        # 5. Final LayerNorm
        if getattr(scorer.hf_model.config, "bottom_final_norm", True):
            logger.info("Seeding bottom final_norm from backbone")
            scorer.hf_model.model.final_norm.load_state_dict(
                full_backbone.model.final_norm.state_dict()
            )

        # 6. Classifier / Head
        if hasattr(full_backbone, "head"):
            logger.info("Seeding head from backbone")
            if hasattr(scorer.hf_model, "head"):
                scorer.hf_model.head.load_state_dict(full_backbone.head.state_dict())
            elif hasattr(scorer.hf_model, "classifier") and hasattr(
                full_backbone.head, "dense"
            ):
                # Copy head dense weight if classifier layer matches
                pass

        if self.freeze_bottom_layers:
            scorer.freeze_bottom_layers()

        logger.info("Successfully initialized ModernBertMiceScorer weights.")


def find_latest_epoch_dir(ckpt_path: Path) -> Optional[Path]:
    """Finds the latest epoch checkpoint directory (e.g. epoch-00000150) containing trainer.pth & optimizer.pth."""
    search_dirs = []
    checkpoints_dir = next(
        (p for p in [ckpt_path, *ckpt_path.parents] if p.name == "checkpoints"),
        None,
    )
    if checkpoints_dir is not None:
        search_dirs.append(checkpoints_dir)
    search_dirs.extend([ckpt_path, ckpt_path.parent])

    for d in search_dirs:
        if not d.exists():
            continue
        epoch_dirs = []
        for f in d.glob("epoch-*"):
            if f.is_dir():
                try:
                    epoch_num = int(f.name[len("epoch-") :])
                    epoch_dirs.append((epoch_num, f))
                except ValueError:
                    pass
        if epoch_dirs:
            epoch_dirs.sort(key=lambda x: x[0], reverse=True)
            for _, epoch_dir in epoch_dirs:
                if (epoch_dir / "trainer.pth").exists() and (
                    epoch_dir / "optimizer.pth"
                ).exists():
                    return epoch_dir
    return None


def is_rank_zero() -> bool:
    """Returns True if this is global rank 0 or running in a single-process environment."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    for env_var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(env_var)
        if val is not None:
            try:
                return int(val) == 0
            except ValueError:
                pass
    return True


def get_rank(fabric=None) -> int:
    """Returns global rank integer (0 if single-process or uninitialized)."""
    if fabric is not None and getattr(fabric, "global_rank", None) is not None:
        return int(fabric.global_rank)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    for env_var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(env_var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


class InitMICEV2FromCheckpoint(LightweightTask):
    """Initializes MICE V2 model from a trained Mice checkpoint path/loader, offers various other options e.g freezing bottom layers & utility predictor."""

    scorer: Param["ModernBertMiceScorer"]

    freeze_bottom_layers: Param[bool] = field(default=True, ignore_default=True)
    """Whether to freeze the bottom contextualization layers and embeddings after loading checkpoint"""

    checkpoint_loader: Param[ModuleLoader]
    """Path or ModuleLoader to the Phase 1 trained checkpoint (should contain model.safetensors or model/ directory)."""

    freeze_utility: Param[bool] = field(default=True, ignore_default=True)
    """Whether to freeze the utility predictor after loading checkpoint"""

    freeze_top_layers: Param[bool] = field(default=False, ignore_default=True)
    """Whether to freeze the top interaction layers and head after loading checkpoint (leaving bottleneck trainable)"""

    def execute(self):
        ckpt_path = Path(self.checkpoint_loader.path)
        logger.info("[InitMICEV2] Loading checkpoint from: %s", ckpt_path)

        latest_epoch_dir = find_latest_epoch_dir(ckpt_path)
        is_phase2_continuation = latest_epoch_dir is not None

        if (
            is_phase2_continuation
            and getattr(self.scorer, "compress_init_option", "seed") == "random"
        ):
            raise ValueError(
                "Cannot continue trainer checkpoint (epoch directory) with compress_init_option='random'. "
                "Either start a new training run from an exported model or use compress_init_option='seed'."
            )

        if is_phase2_continuation:
            # 1. Initialize scorer and freeze parameters on all ranks
            if not self.scorer._initialized:
                self.scorer.initialize()

            if self.freeze_utility:
                self.scorer.freeze_utility()

            if self.freeze_bottom_layers:
                self.scorer.freeze_bottom_layers()

            if self.freeze_top_layers:
                self.scorer.freeze_top_layers()

            # 2. Checkpoint copying only on rank zero for epoch directories
            if not is_rank_zero():
                logger.info(
                    "[InitMICEV2] Skipping checkpoint copying on non-zero rank (RANK=%s, LOCAL_RANK=%s)",
                    os.environ.get("RANK", "unknown"),
                    os.environ.get("LOCAL_RANK", "unknown"),
                )
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.barrier()
                return
            else:
                logger.info(
                    "[InitMICEV2] Found full checkpoint from trainer path at: %s. "
                    "Copying full checkpoint (model weights, optimizer.pth, trainer.pth) to epoch-00000000.",
                    latest_epoch_dir,
                )

            # Read Phase 1 step count from info.json
            phase1_steps = 0
            if (latest_epoch_dir / "info.json").exists():
                try:
                    with (latest_epoch_dir / "info.json").open("rt") as fp:
                        p1_info = json.load(fp)
                        phase1_steps = p1_info.get("steps", 0)
                    logger.info(
                        "[InitMICEV2] Preserving Phase 1 step count (%d) for TensorBoard continuity at epoch 0",
                        phase1_steps,
                    )
                except Exception as e:
                    logger.warning(
                        "[InitMICEV2] Could not read steps from Phase 1 info.json: %s",
                        e,
                    )

            # Helper function to populate epoch-00000000 directory
            def copy_to_epoch_zero(target_base_dir: Path):
                epoch_zero = target_base_dir / "epoch-00000000"
                if epoch_zero.exists():
                    logging.warning(
                        "[InitMICEV2] Removing existing epoch-00000000 directory at %s",
                        epoch_zero,
                    )
                    shutil.rmtree(epoch_zero)

                shutil.copytree(latest_epoch_dir, epoch_zero)

                # Copy trainer & optimizer state dicts if present
                if (latest_epoch_dir / "trainer.pth").exists():
                    shutil.copy2(
                        latest_epoch_dir / "trainer.pth", epoch_zero / "trainer.pth"
                    )
                # Write info.json resetting epoch to 0 but preserving total steps
                with (epoch_zero / "info.json").open("wt") as fp:
                    json.dump({"epoch": 0, "steps": phase1_steps}, fp)

                logger.info("[InitMICEV2] Populated epoch-00000000 at %s", epoch_zero)

            # Populate process cwd checkpoints/epoch-00000000
            copy_to_epoch_zero(Path.cwd() / "checkpoints")

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        # latest_epoch_dir is None: exported model or HF Hub checkpoint
        model_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
        if (
            (model_dir / "model.safetensors").exists()
            or (model_dir / "pytorch_model.bin").exists()
            or (model_dir / "model").exists()
        ):
            target_compress_dim = getattr(self.scorer, "compress_dim", 1.0)
            target_compress_init = getattr(self.scorer, "compress_init_option", "seed")

            ckpt_compress_dim = 1.0
            cfg_path = model_dir / "config.json"
            if not cfg_path.exists() and (model_dir / "model" / "config.json").exists():
                cfg_path = model_dir / "model" / "config.json"
            if cfg_path.exists():
                try:
                    with cfg_path.open("rt") as f:
                        ckpt_cfg = json.load(f)
                        ckpt_compress_dim = ckpt_cfg.get("compress_dim", 1.0)
                except Exception as e:
                    logger.warning(
                        "[InitMICEV2] Failed to read config.json at %s: %s", cfg_path, e
                    )

            ignore_top = (
                target_compress_dim != ckpt_compress_dim
                or target_compress_init == "random"
            )

            logger.info(
                "[InitMICEV2] latest_epoch_dir is None: loading weights directly from exported model %s (ignore_top=%s)",
                model_dir,
                ignore_top,
            )
            self.scorer.load_model(model_dir, ignore_top=ignore_top)

            target_compress_dim = getattr(self.scorer, "compress_dim", 1.0)
            target_compress_init = getattr(self.scorer, "compress_init_option", "seed")
            current_compress_dim = getattr(
                self.scorer.hf_model.config, "compress_dim", 1.0
            )

            if (
                target_compress_dim != current_compress_dim
                or target_compress_init == "random"
            ):
                logger.info(
                    "[InitMICEV2] Calling build_top_layers: "
                    "compress_dim=%s (was %s), compress_init_option=%r",
                    target_compress_dim,
                    current_compress_dim,
                    target_compress_init,
                )
                self.scorer.build_top_layers(
                    compress_dim=target_compress_dim,
                    compress_init_option=target_compress_init,
                )

            if self.freeze_utility:
                self.scorer.freeze_utility()
            if self.freeze_bottom_layers:
                self.scorer.freeze_bottom_layers()
            if self.freeze_top_layers:
                self.scorer.freeze_top_layers()
        else:
            logger.info(
                "[InitMICEV2] latest_epoch_dir is None and no exported weights file found at %s. "
                "Skipping weight loading.",
                ckpt_path,
            )
            if not self.scorer._initialized:
                self.scorer.initialize()
            if self.freeze_utility:
                self.scorer.freeze_utility()
            if self.freeze_bottom_layers:
                self.scorer.freeze_bottom_layers()
            if self.freeze_top_layers:
                self.scorer.freeze_top_layers()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


class EvalSparsityAccumulator:
    """Accumulator for evaluation document token sparsity metrics across batches/ranks."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.total_tokens: int = 0
        self.pruned_tokens: int = 0
        self.sum_utilities: float = 0.0
        logger.info(
            f"[EvalSparsityAccumulator] Instantiated with threshold={self.threshold}"
        )

    def reset(self):
        logger.debug("[EvalSparsityAccumulator] Resetting token accumulators")
        self.total_tokens = 0
        self.pruned_tokens = 0
        self.sum_utilities = 0.0

    def update(self, utilities: torch.Tensor, doc_mask: Optional[torch.Tensor] = None):
        """Accumulate token counts and utilities for non-padding tokens."""
        if utilities is None or utilities.numel() == 0:
            return

        if (
            doc_mask is not None
            and utilities.ndim == 3
            and utilities.shape[:2] == doc_mask.shape[:2]
        ):
            mask = doc_mask.unsqueeze(-1).bool()
            valid_u = utilities.masked_select(mask)
        else:
            valid_u = utilities.view(-1)

        if valid_u.numel() == 0:
            return

        n_tokens = valid_u.numel()
        n_pruned = int((valid_u < self.threshold).sum().item())
        sum_u = float(valid_u.sum().item())

        self.total_tokens += n_tokens
        self.pruned_tokens += n_pruned
        self.sum_utilities += sum_u

    def compute(self, fabric=None) -> Tuple[Dict[str, float], bool]:
        """Aggregate stats across Fabric ranks or torch.distributed (if multi-GPU) and return metrics.

        Returns:
            (stats, gather_succeeded): Tuple of metric dictionary and a boolean
            indicating whether multi-rank gathering succeeded.
        """
        total = self.total_tokens
        pruned = self.pruned_tokens
        sum_u = self.sum_utilities

        is_multi_rank = (
            fabric is not None and getattr(fabric, "world_size", 1) > 1
        ) or (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        gather_succeeded = not is_multi_rank

        if is_multi_rank:
            try:
                if not (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    raise RuntimeError(
                        "torch.distributed is uninitialized or already destroyed at exit"
                    )

                if fabric is not None and getattr(fabric, "world_size", 1) > 1:
                    t = torch.tensor(
                        [total, pruned, sum_u],
                        device=fabric.device,
                        dtype=torch.float64,
                    )
                    t = fabric.all_reduce(t, reduce_op="sum")
                    total = int(t[0].item())
                    pruned = int(t[1].item())
                    sum_u = float(t[2].item())
                    gather_succeeded = True
                    logger.info(
                        f"[EvalSparsityAccumulator] Fabric all-reduced stats: total={total}, pruned={pruned}, sum={sum_u:.4f}"
                    )
                elif (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    t = torch.tensor(
                        [total, pruned, sum_u],
                        dtype=torch.float64,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                    total = int(t[0].item())
                    pruned = int(t[1].item())
                    sum_u = float(t[2].item())
                    gather_succeeded = True
                    logger.info(
                        f"[EvalSparsityAccumulator] torch.distributed all-reduced stats: total={total}, pruned={pruned}, sum={sum_u:.4f}"
                    )
            except Exception as e:
                gather_succeeded = False
                logger.warning(
                    f"[EvalSparsityAccumulator] Multi-rank gather failed ({e}). "
                    f"Falling back to local rank metrics (total_tokens={total})."
                )

        denom = max(total, 1)
        stats = {
            "sparsity": pruned / denom,
            "mean_utility": sum_u / denom,
            "kept_ratio": (total - pruned) / denom,
            "total_tokens": total,
            "pruned_tokens": pruned,
            "sum_utilities": sum_u,
        }
        logger.info(
            f"[EvalSparsityAccumulator] Computed metrics (gather_succeeded={gather_succeeded}): {stats}"
        )
        return stats, gather_succeeded

    def flush_to_file(self, fabric=None, output_filename: str = "eval_sparsity.json"):
        """Flush computed metrics to JSON file.

        - Rank 0: Always writes output_filename (default: eval_sparsity.json).
        - Rank != 0: Only writes eval_sparsity_{rank}.json if multi-rank gather failed.
        """
        rank = get_rank(fabric=fabric)
        logger.info(
            f"[EvalSparsityAccumulator] Triggering flush_to_file at CWD={Path.cwd()} (rank={rank}, total_tokens={self.total_tokens})"
        )
        if self.total_tokens == 0:
            logger.warning(
                f"[EvalSparsityAccumulator] Rank {rank}: total_tokens is 0 during flush_to_file; skipping write."
            )
            return

        stats, gather_succeeded = self.compute(fabric=fabric)

        if rank == 0:
            out_path = Path.cwd() / output_filename
            try:
                with out_path.open("w") as fp:
                    json.dump(stats, fp, indent=2)
                logger.info(
                    f"[EvalSparsityAccumulator] (Rank 0) Successfully wrote evaluation sparsity metrics to {out_path}: {stats}"
                )
            except Exception as e:
                logger.error(
                    f"[EvalSparsityAccumulator] Failed to write {output_filename}: {e}"
                )
        elif not gather_succeeded:
            out_filename = f"eval_sparsity_{rank}.json"
            out_path = Path.cwd() / out_filename
            try:
                with out_path.open("w") as fp:
                    json.dump(stats, fp, indent=2)
                logger.warning(
                    f"[EvalSparsityAccumulator] (Rank {rank}) Wrote fallback per-rank sparsity metrics to {out_path}: {stats}"
                )
            except Exception as e:
                logger.error(
                    f"[EvalSparsityAccumulator] Failed to write {out_filename}: {e}"
                )


class InitEvalSparsityHook(LightweightTask):
    """Init task attached to evaluations that initializes EvalSparsityAccumulator on ModernBertMiceScorer and registers an atexit hook to flush eval_sparsity.json at evaluation process exit."""

    scorer: Param["ModernBertMiceScorer"]
    threshold: Param[Optional[float]] = field(default=None, ignore_default=True)

    def execute(self):
        logger.debug(
            f"[InitEvalSparsityHook] Lifecycle Step: Executing InitTask in CWD={Path.cwd()}"
        )
        scorer = self.scorer.module if hasattr(self.scorer, "module") else self.scorer

        effective_threshold = self.threshold
        if effective_threshold is None:
            effective_threshold = getattr(scorer, "hard_prune_threshold", 0.5)
            if hasattr(scorer, "hf_model") and hasattr(scorer.hf_model, "config"):
                effective_threshold = getattr(
                    scorer.hf_model.config, "hard_prune_threshold", effective_threshold
                )

            logger.info(
                f"[InitEvalSparsityHook] Effective eval sparsity threshold automatically resolved to: {effective_threshold}"
            )

        if (
            not hasattr(scorer, "eval_sparsity_tracker")
            or scorer.eval_sparsity_tracker is None
        ):
            logger.debug(
                f"[InitEvalSparsityHook] Attaching new EvalSparsityAccumulator(threshold={effective_threshold}) to scorer"
            )
            scorer.eval_sparsity_tracker = EvalSparsityAccumulator(
                threshold=effective_threshold
            )
        else:
            logger.debug(
                f"[InitEvalSparsityHook] Updating existing EvalSparsityAccumulator threshold to {effective_threshold}"
            )
            scorer.eval_sparsity_tracker.threshold = effective_threshold

        scorer.eval_sparsity_tracker.reset()

        def _atexit_flush():
            logger.info(
                "[InitEvalSparsityHook] Lifecycle Step: atexit handler triggered at process exit"
            )
            fabric = getattr(scorer, "fabric", None)
            if (
                hasattr(scorer, "eval_sparsity_tracker")
                and scorer.eval_sparsity_tracker is not None
            ):
                scorer.eval_sparsity_tracker.flush_to_file(fabric=fabric)
            else:
                logger.warning(
                    "[InitEvalSparsityHook] atexit handler found no eval_sparsity_tracker on scorer!"
                )

        atexit.register(_atexit_flush)
        logger.info(
            "[InitEvalSparsityHook] Lifecycle Step: Registered _atexit_flush handler"
        )


class ModernBertMiceScorer(AbstractModuleScorer):
    """
    Experimaestro wrapper for ModernBERT MICE V2 Architecture.
    """

    hf_id: Param[str]
    """Hugging Face checkpoint identifier"""

    tokenizer: Param[MICEQueryDocTokenizer]
    """The tokenizer for independent Q/D processing"""

    n_contextualization_layers: Param[int] = field(default=6, ignore_default=True)
    """Number of bottom encoder layers that process query and document independently"""

    n_docs_ctx_layers: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Number of bottom encoder layers for the document"""

    n_interaction_layers: Param[Optional[int]] = field(
        default=None, ignore_default=True
    )
    """Number of top encoder layers with cross-attention"""

    bound_bottom_layers: Param[bool] = field(default=True, ignore_default=True)
    """Whether to bound query and document encoding layers"""

    attn_implementation: Param[Optional[str]] = field(default=None, ignore_default=True)
    """Attention implementation (e.g. 'flash_attention_2', 'sdpa', 'eager')"""

    utility_pred: Param[bool] = field(default=False, ignore_default=True)
    """Whether to add a 2layer MLP utility predictor applied to the document hidden states before upper layers"""

    hard_pruning: Param[bool] = field(default=False, ignore_default=True)
    """Whether to hard-prune tokens with utility < hard_prune_threshold in Phase 2 training"""

    hard_prune_threshold: Param[float] = field(default=0.5, ignore_default=True)
    """Utility threshold below which tokens are hard-pruned in Phase 2 training (default 0.5)"""

    normalize_doc_embeds: Param[bool] = field(default=False, ignore_default=True)
    """Whether to L2-normalize document hidden states before utility prediction and upper interaction layers"""

    use_upper_rope: Param[bool] = field(default=True, ignore_default=True)
    """Whether to use Rotary Position Embedding (RoPE) for query tokens in upper interaction layers (disabled for documents)"""

    top_layers_doc_ln: Param[bool] = field(default=False, ignore_default=True)
    """Whether to use a document LayerNorm in top interaction layers and multiply token utilities post-LayerNorm"""

    bottom_final_norm: Param[bool] = field(default=True, ignore_default=True)
    """Whether to apply LayerNorm at the output of bottom contextualization layers"""

    top_final_norm: Param[bool] = field(default=False, ignore_default=True)
    """Whether to apply LayerNorm at the output of top interaction layers before classifier"""

    utility_temp: Param[float] = field(default=1.0, ignore_default=True)
    """Initial temperature for utility predictor sigmoid"""

    utility_bias: Param[float] = field(default=5.0, ignore_default=True)
    """Initial bias scalar for utility predictor output projection wo.bias"""

    gate_target: Param[str] = field(default="h", ignore_default=True)
    """Target representation for gating: 'h' (default, gate full document hidden states) or 'v' (gate only cross-attention Value vectors)"""

    use_hard_concrete: Param[bool] = field(default=False, ignore_default=True)
    """Whether to use Hard Concrete L0 distribution sampling for utility prediction"""

    hc_gamma: Param[float] = field(default=-0.1, ignore_default=True)
    """Hard concrete lower stretch parameter gamma (default -0.1)"""

    hc_zeta: Param[float] = field(default=1.1, ignore_default=True)
    """Hard concrete upper stretch parameter zeta (default 1.1)"""

    compress_dim: Param[float] = field(default=1.0, ignore_default=True)
    """Compression ratio (>=1.0) applied to hidden_size/intermediate_size/num_attention_heads of the top interaction layers only"""

    compress_init_option: Param[str] = field(default="seed", ignore_default=True)
    """How top interaction layers are initialized: 'seed' (load from the pretrained
    checkpoint when compress_dim == 1.0, else falls back to 'random'), 'random' (always
    skip seeding, keep each layer's own default init), 'random_projection', or
    'svd_projection' (both only meaningful when compress_dim > 1.0)"""

    use_cross_attn: Param[bool] = field(default=True, ignore_default=True)
    """Whether top layers use Cross+Self attention (True) or Concatenated Self-Attention only (False)"""

    head_type: Param[str] = field(default="linear", ignore_default=True)
    """Type of classification head pipeline: 'linear' (default, legacy single linear layer) or 'st_reranker' (ST pooling + dense + LayerNorm + dense)"""

    doc_bottleneck_dim: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Bottleneck dimension for document token embeddings (d). If set, compresses document embeddings to d dimensions before storage/cross-attention."""

    _version: Constant[str] = "2.0.2"

    hf_model: nn.Module

    def freeze_utility(self):
        """Freezes utility predictor."""
        self.hf_model.freeze_utility()

    def freeze_bottom_layers(self):
        """Freezes bottom layers and embeddings."""
        self.hf_model.freeze_bottom_layers()

    def freeze_top_layers(self):
        """Freezes top interaction layers, rotary embeddings, adapter, and classification head."""
        self.hf_model.freeze_top_layers()

    def build_top_layers(
        self,
        compress_dim: Optional[float] = None,
        compress_init_option: Optional[str] = None,
        n_interaction_layers: Optional[int] = None,
        head_type: Optional[str] = None,
        doc_bottleneck_dim: Optional[int] = None,
    ):
        """Rebuilds top interaction layers on self.hf_model and synchronizes scorer parameters."""
        if not self._initialized:
            self.initialize()
        self.hf_model.build_top_layers(
            compress_dim=compress_dim,
            compress_init_option=compress_init_option,
            n_interaction_layers=n_interaction_layers,
            head_type=head_type,
            doc_bottleneck_dim=doc_bottleneck_dim,
        )
        if compress_dim is not None:
            self.compress_dim = compress_dim
        if compress_init_option is not None:
            self.compress_init_option = compress_init_option
        if n_interaction_layers is not None:
            self.n_interaction_layers = n_interaction_layers
        if head_type is not None:
            self.head_type = head_type
        if doc_bottleneck_dim is not None:
            self.doc_bottleneck_dim = doc_bottleneck_dim

    def apply_config_overrides(
        self,
        mice_config: ModernBertMiceConfig,
        source_path_or_id: Union[str, Path],
        check_attrs: Optional[List[str]] = None,
    ):
        """Applies parameters from self (the xpm Config source of truth) to mice_config,
        logging loud [MICE Config Override] warnings whenever a parameter differs from the checkpoint."""
        if check_attrs is None:
            check_attrs = [
                "n_contextualization_layers",
                "n_docs_ctx_layers",
                "n_interaction_layers",
                "bound_bottom_layers",
                "utility_pred",
                "hard_pruning",
                "hard_prune_threshold",
                "normalize_doc_embeds",
                "use_upper_rope",
                "top_layers_doc_ln",
                "bottom_final_norm",
                "top_final_norm",
                "utility_temp",
                "utility_bias",
                "gate_target",
                "use_hard_concrete",
                "hc_gamma",
                "hc_zeta",
                "compress_dim",
                "compress_init_option",
                "doc_bottleneck_dim",
                "use_cross_attn",
                "head_type",
            ]

        for attr in check_attrs:
            if not hasattr(mice_config, attr):
                continue
            self_val = getattr(self, attr, None)
            if self_val is None or isinstance(self_val, field):
                continue
            ckpt_val = getattr(mice_config, attr)
            if self_val != ckpt_val:
                logger.warning(
                    f"[MICE Config Override] Overriding checkpoint parameter '{attr}' "
                    f"from '{ckpt_val}' (saved in {source_path_or_id}/config.json) to '{self_val}' "
                    f"(specified on {self.__class__.__name__}). "
                    "This creates an architectural discrepancy with pre-trained weights and may require retraining!"
                )
                setattr(mice_config, attr, self_val)

        if self.attn_implementation is not None:
            ckpt_attn = getattr(mice_config, "_attn_implementation", None)
            if self.attn_implementation != ckpt_attn:
                logger.info(
                    f"[MICE Config] Setting _attn_implementation to {self.attn_implementation!r} "
                    f"(was {ckpt_attn!r} in {source_path_or_id})"
                )
                mice_config._attn_implementation = self.attn_implementation

    def __initialize__(self):
        super().__initialize__()
        logger.info("Initializing ModernBertMiceCrossEncoder")
        self.eval_sparsity_tracker: Optional[EvalSparsityAccumulator] = None
        self.tokenizer.initialize()

        loaded_config = AutoConfig.from_pretrained(self.hf_id)
        is_mice_model = (
            isinstance(loaded_config, ModernBertMiceConfig)
            or getattr(loaded_config, "model_type", None) == "modernbert_mice"
        )
        if is_mice_model:
            logger.info(f"Loaded existing MICE config from {self.hf_id}")
            mice_config = loaded_config

            self.apply_config_overrides(
                mice_config,
                source_path_or_id=self.hf_id,
            )
        else:
            logger.info(f"Building ModernBertMiceConfig from backbone {self.hf_id}")
            config_kwargs = loaded_config.to_dict()
            if self.attn_implementation is not None:
                config_kwargs["_attn_implementation"] = self.attn_implementation

            mice_config = ModernBertMiceConfig(
                n_contextualization_layers=self.n_contextualization_layers,
                n_docs_ctx_layers=self.n_docs_ctx_layers,
                n_interaction_layers=self.n_interaction_layers,
                bound_bottom_layers=self.bound_bottom_layers,
                utility_pred=self.utility_pred,
                hard_pruning=self.hard_pruning,
                hard_prune_threshold=self.hard_prune_threshold,
                normalize_doc_embeds=self.normalize_doc_embeds,
                use_upper_rope=self.use_upper_rope,
                top_layers_doc_ln=self.top_layers_doc_ln,
                bottom_final_norm=self.bottom_final_norm,
                top_final_norm=self.top_final_norm,
                utility_temp=self.utility_temp,
                utility_bias=self.utility_bias,
                gate_target=self.gate_target,
                use_hard_concrete=self.use_hard_concrete,
                hc_gamma=self.hc_gamma,
                hc_zeta=self.hc_zeta,
                compress_dim=self.compress_dim,
                compress_init_option=self.compress_init_option,
                doc_bottleneck_dim=self.doc_bottleneck_dim,
                use_cross_attn=self.use_cross_attn,
                head_type=self.head_type,
                **config_kwargs,
            )

        if is_mice_model:
            logger.info(f"Loading pretrained MICE weights from {self.hf_id}")
            hf_model = ModernBertMiceForSequenceClassification.from_pretrained(
                self.hf_id, config=mice_config
            )
        else:
            hf_model = ModernBertMiceForSequenceClassification(mice_config)

        self.add_module("hf_model", hf_model)

        attn_impl = getattr(hf_model.config, "_attn_implementation", "sdpa")
        if "flash" in str(attn_impl).lower():
            logger.info(
                f"Using Flash Attention 2 for MICE V2 ModernBERT (`{attn_impl}`)."
            )
        else:
            logger.warning(
                f"Flash Attention 2 is not enabled or available (using `{attn_impl}`). "
                "For optimal performance and unpadded attention, set `_attn_implementation='flash_attention_2'`."
            )

    def setup_with_fabric(self, fabric) -> torch.nn.Module:
        """Sets up the module with PyTorch Lightning Fabric, checking precision compatibility.

        If Fabric runs in float32 precision (precision is None or '32-true'), FlashAttention-2
        cannot execute and is automatically downgraded to 'sdpa'.

        Args:
            fabric: The PyTorch Lightning Fabric instance.

        Returns:
            torch.nn.Module: The Fabric-wrapped module proxy.
        """
        # extra check for FlashAttention-2 compatibility and fallback to SDPA if needed.
        fallback_fa2_if_incompatible_precision(self.hf_model, fabric)
        return super().setup_with_fabric(fabric)

    def encode_documents(
        self, doc_ids: torch.Tensor, doc_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode documents through bottom contextualization layers.

        Returns (d_hidden, doc_mask), see `ModernBertMiceForSequenceClassification.encode_documents`.
        """
        return self.hf_model.encode_documents(doc_ids=doc_ids, doc_mask=doc_mask)

    @property
    def model(self) -> torch.nn.Module:
        """Underlying torch module (self.hf_model) for Fabric FA2 compatibility."""
        return self.hf_model

    @property
    def dimension(self) -> int:
        """Token embedding dimension of bottom contextualization layers."""
        doc_bot_dim = getattr(self.hf_model.config, "doc_bottleneck_dim", None)
        if doc_bot_dim is not None and doc_bot_dim > 0:
            return doc_bot_dim
        top_config = getattr(self.hf_model, "top_config", self.hf_model.config)
        return getattr(top_config, "hidden_size", 768)

    def document_token_embeddings(
        self, records: List[IDTextRecord], max_doc_length: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Encode a batch of documents and return per-token embeddings filtering padding out.

        Matches the MultiVectorEncoder duck-typed protocol for PlaidIndexBuilder.
        """
        effective_max_doc_len = (
            max_doc_length
            if max_doc_length is not None
            else self.tokenizer.max_doc_length
        )
        options = TokenizerOptions(max_length=effective_max_doc_len)
        tokenized = self.batch_tokenize(
            records, options=options, input_nature=QueryDocInput.DOCUMENT
        )
        if tokenized.ids.device != self.device:
            tokenized = tokenized.to(self.device)
        encoded, doc_mask = self.encode_documents(tokenized.ids, tokenized.mask)
        # Apply pruning mask if hard_pruning / utility gating is enabled and shapes differ
        if doc_mask is not None and doc_mask.shape[1] != encoded.shape[1]:
            mask = doc_mask.bool()
        elif tokenized.mask is not None:
            mask = tokenized.mask.to(encoded.device).bool()
        else:
            mask = None
        if mask is None:
            return [encoded[i] for i in range(encoded.shape[0])]
        return [encoded[i][mask[i]] for i in range(encoded.shape[0])]

    def get_document_encoder(
        self, max_doc_length: Optional[int] = None
    ) -> TextEncoderBase:
        """Returns a document encoder for fast-plaid index building."""
        model_cfg = getattr(self, "config", self)
        return MiceV2DocumentEncoder.C(model=model_cfg, max_doc_length=max_doc_length)

    def get_query_encoder(
        self, max_query_length: Optional[int] = None
    ) -> AbstractModuleScorer:
        """Returns a query encoder for fast-plaid search, optionally overriding max_query_length."""
        if max_query_length is None:
            return self
        model_cfg = getattr(self, "config", self)
        return MiceV2QueryEncoder.C(model=model_cfg, max_query_length=max_query_length)

    def encode_queries(
        self, query_ids: torch.Tensor, query_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode queries through bottom contextualization layers.

        Returns (q_hidden, query_mask), see `ModernBertMiceForSequenceClassification.encode_queries`.
        """
        return self.hf_model.encode_queries(query_ids=query_ids, query_mask=query_mask)

    def query_token_embeddings(
        self, records: List[IDTextRecord], max_query_length: Optional[int] = None
    ) -> torch.Tensor:
        """Returns a dense (batch, query_maxlen, dim) query embedding tensor
        for genuine ColBERT-style search via fast-plaid, with padding
        positions zeroed out so they contribute ~0 to MaxSim regardless of
        document token (MICE's bottom layers were never trained with
        ColBERT's [MASK]-token query-augmentation convention, so we don't
        attempt to replicate it here).

        This makes `ModernBertMiceScorer` itself directly usable as
        `xpmir.index.plaid.PlaidRetriever.encoder` (it's already an
        `AbstractModuleScorer`, which is what that Param expects).
        """
        effective_max_q_len = (
            max_query_length
            if max_query_length is not None
            else self.tokenizer.max_query_length
        )
        options = TokenizerOptions(max_length=effective_max_q_len)
        tokenized = self.batch_tokenize(
            records, options=options, input_nature=QueryDocInput.QUERY
        )
        if tokenized.ids.device != self.device:
            tokenized = tokenized.to(self.device)
        q_hidden, q_mask = self.encode_queries(tokenized.ids, tokenized.mask)
        return q_hidden * q_mask.unsqueeze(-1).to(q_hidden.dtype)

    def batch_tokenize(
        self,
        input_records: BaseItems,
        options=None,
        input_nature: QueryDocInput = QueryDocInput.PAIRS,
    ) -> MICETokenizedTexts:
        return self.tokenizer.tokenize(
            input_records, options=options, input_nature=input_nature
        )

    def get_tokenizer_fn(self):
        return self.batch_tokenize

    def forward(
        self,
        inputs: Optional[BaseItems] = None,
        tokenized: Optional[MICETokenizedTexts] = None,
        doc_hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if inputs is None and tokenized is None:
            raise ValueError("Either raw inputs or tokenized inputs must be provided.")

        if tokenized is None:
            tokenized = self.batch_tokenize(inputs, input_nature=QueryDocInput.PAIRS)

        tokenized_q = to_device(tokenized.tokenized_q, self.device)

        if doc_hidden_states is not None:
            # Decompressed document token embeddings from fast-plaid index (list of 2D tensors [doc_len, dim])
            doc_tensors = [to_device(d, self.device) for d in doc_hidden_states]
            max_d_len = max(d.shape[0] for d in doc_tensors)
            batch_size = len(doc_tensors)
            dim = doc_tensors[0].shape[-1]

            doc_hidden = torch.zeros(
                (batch_size, max_d_len, dim),
                dtype=doc_tensors[0].dtype,
                device=self.device,
            )
            doc_mask = torch.zeros(
                (batch_size, max_d_len),
                dtype=torch.long,
                device=self.device,
            )
            for i, d in enumerate(doc_tensors):
                seq_len = d.shape[0]
                doc_hidden[i, :seq_len] = d
                doc_mask[i, :seq_len] = 1

            out = self.hf_model(
                query_ids=tokenized_q.ids,
                query_mask=tokenized_q.mask,
                doc_hidden=doc_hidden,
                doc_mask=doc_mask,
            )
            target_doc_mask = doc_mask
        else:
            tokenized_docs = to_device(tokenized.tokenized_docs, self.device)
            out = self.hf_model(
                query_ids=tokenized_q.ids,
                query_mask=tokenized_q.mask,
                doc_ids=tokenized_docs.ids,
                doc_mask=tokenized_docs.mask,
            )
            target_doc_mask = tokenized_docs.mask

        info = TrainerContext.current()

        if info is not None:
            if getattr(out, "utilities", None) is not None:
                for hook in info.hooks(UtilityRegularizer):
                    hook(
                        info,
                        out.utilities,
                        doc_mask=target_doc_mask,
                        p_active=getattr(out, "p_active", None),
                        doc_hidden=getattr(out, "doc_hidden", None),
                        hf_model=self.hf_model,
                    )

                for hook in info.hooks(GatedValueRegularizer):
                    hook(
                        info,
                        self.hf_model,
                        out.utilities,
                        getattr(out, "doc_hidden", None),
                        doc_mask=target_doc_mask,
                    )
        elif self.training and getattr(out, "utilities", None) is not None:
            if not getattr(self, "_warned_missing_context", False):
                logger.warning(
                    "[ModernBertMiceScorer] Model is in training mode (self.training=True) and "
                    "utilities were computed, but no active TrainerContext was found via "
                    "TrainerContext.current(). Regularization hooks (UtilityRegularizer, "
                    "GatedValueRegularizer) will NOT execute! "
                    "Wrap your execution block in 'with context:' or 'with context.losses():'."
                )
                self._warned_missing_context = True

        if (
            not self.training
            and getattr(self, "eval_sparsity_tracker", None) is not None
            and getattr(out, "utilities", None) is not None
        ):
            logger.debug(
                "[ModernBertMiceScorer] Updating eval_sparsity_tracker during evaluation forward pass"
            )
            self.eval_sparsity_tracker.update(out.utilities, doc_mask=target_doc_mask)

        return out.logits.squeeze(-1)

    def hub_readme_sections(self) -> List["ReadmeSection"]:
        """Returns structured README sections (main card template + separate evaluations section)."""
        from xpm_torch.module import ReadmeSection
        from xpm_torch.utils import count_safetensors_params, format_num_params
        from huggingface_hub import ModelCard, ModelCardData

        sections = []
        template_path = Path(__file__).parent / "Micev2Card.md"

        config = getattr(self, "__config__", None)

        export_dir = getattr(self, "_last_export_dir", None) or (
            getattr(config, "_last_export_dir", None) if config else None
        )
        total_params_str = "N/A"
        if export_dir:
            st_path = Path(export_dir) / "model.safetensors"
            if st_path.is_file():
                try:
                    num_params = count_safetensors_params(st_path)
                    total_params_str = (
                        f"{num_params:,} ({format_num_params(num_params)})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not compute parameter count from safetensors: {e}"
                    )

        metadata = (
            getattr(self, "_export_metadata", None)
            or (getattr(config, "_export_metadata", {}) if config else {})
            or {}
        )
        base_name = getattr(self, "hf_id", "ModernBERT")
        model_name = (
            getattr(self, "model_name", None)
            or (getattr(config, "model_name", None) if config else None)
            or (Path(export_dir).name if export_dir else "MICE v2 Model")
        )

        cfg = (
            getattr(self.hf_model, "config", None)
            if hasattr(self, "hf_model")
            else None
        )
        n_ctx = getattr(
            cfg,
            "n_contextualization_layers",
            getattr(self, "n_contextualization_layers", 2),
        )
        n_inter = getattr(
            cfg,
            "n_interaction_layers",
            getattr(self, "n_interaction_layers", None),
        )
        if n_inter is None and cfg is not None:
            n_inter = getattr(cfg, "num_hidden_layers", 0) - n_ctx

        # 1. Main Model Card Section (from Micev2Card.md template)
        if template_path.exists():
            card_data = ModelCardData(
                language="en",
                license="apache-2.0",
                base_model=str(base_name),
                model_name=str(model_name),
                pipeline_tag="text-classification",
            )
            training_strat = metadata.get("training_strat") or metadata.get(
                "loss", "N/A"
            )
            card = ModelCard.from_template(
                card_data,
                template_path=str(template_path),
                base=str(base_name),
                model_id=str(model_name),
                total_parameters=total_params_str,
                n_contextualization_layers=n_ctx,
                n_interaction_layers=n_inter,
                training_strat=training_strat,
                loss=metadata.get("loss", "N/A"),
            )
            sections.append(ReadmeSection(key="model_card", content=card.content))

        # 2. Separate Evaluations Section (positioned after model_card)
        results_md = metadata.get("results")
        if results_md:
            retriever = metadata.get("retriever") or "naver/splade-v3-distilbert"
            k = metadata.get("k") or 100
            if k == "N/A":
                k = 100
            eval_content = f"## Evaluations\n\nWe provide evaluations of this cross-encoder re-ranking top {k} of `{retriever}`:\n\n{results_md}"
            sections.append(
                ReadmeSection(
                    key="evaluations",
                    after="model_card",
                    content=eval_content,
                )
            )

        return sections

    def write_hub_extras(
        self,
        save_directory: Path,
        tb_path: Optional[Union[str, Path]] = None,
    ):
        """Called by TorchHFHub during model export."""

        save_path = Path(save_directory)
        self._last_export_dir = save_path
        self.save_model(save_path)
        self._write_st_metadata(save_path)

        # Copy TensorBoard log directory if available
        tb_dir = (
            Path(tb_path)
            if tb_path
            else getattr(self, "_tb_path", None)
            or (
                Path(self._export_metadata["tb_path"])
                if hasattr(self, "_export_metadata")
                and isinstance(self._export_metadata, dict)
                and self._export_metadata.get("tb_path")
                else None
            )
        )

        if tb_dir and Path(tb_dir).exists():
            target_tb_dir = save_path / "runs"
            try:
                if target_tb_dir.exists():
                    shutil.rmtree(target_tb_dir)
                shutil.copytree(tb_dir, target_tb_dir, symlinks=False)
                logger.info(f"Copied TensorBoard logs from {tb_dir} to {target_tb_dir}")
            except Exception as e:
                logger.warning(
                    f"Failed to copy TensorBoard logs to {target_tb_dir}: {e}"
                )
        else:
            logger.warning(
                f"No TensorBoard log directory found for model export (searched tb_path={tb_path}, _tb_path={getattr(self, '_tb_path', None)}, export_dir={self._last_export_dir})"
            )

    def _write_st_metadata(self, save_directory: Path):
        """Write native SentenceTransformers metadata files for CrossEncoder compatibility."""
        import json

        save_directory = Path(save_directory)

        # 1. modules.json
        modules = [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.base.modules.transformer.Transformer",
            }
        ]
        (save_directory / "modules.json").write_text(json.dumps(modules, indent=2))

        # 2. config_sentence_transformers.json
        st_config = {
            "activation_fn": "torch.nn.modules.linear.Identity",
            "default_prompt_name": None,
            "model_type": "CrossEncoder",
            "prompts": {
                "query": "[Q] ",
                "document": "[D] ",
            },
        }
        (save_directory / "config_sentence_transformers.json").write_text(
            json.dumps(st_config, indent=2)
        )

        # 3. sentence_bert_config.json
        # Both "text" and "message" modalities must be declared:
        # "text" allows ST's CrossEncoder input validator to accept raw text pair inputs,
        # while "message" causes ST's Transformer.preprocess to automatically route
        # (query, document) pairs into structured messages formatted by chat_template.jinja.
        sb_config = {
            "transformer_task": "sequence-classification",
            "modality_config": {
                "text": {
                    "method": "forward",
                    "method_output_name": "logits",
                },
                "message": {
                    "method": "forward",
                    "method_output_name": "logits",
                    "format": "flat",
                },
            },
            "module_output_name": "scores",
        }
        (save_directory / "sentence_bert_config.json").write_text(
            json.dumps(sb_config, indent=2)
        )

        # 4. chat_template.jinja
        # ModernBERT BPE whitespace rules:
        # Do not include leading spaces before '[CLS]' or '[Q]' or trailing spaces before '[SEP]',
        # otherwise ModernBERT tokenizes ' [Q]' as token 544 instead of 60 ('['), altering representations.
        chat_template = (
            "{%- set instruction = messages | selectattr('role', 'eq', 'system') | map(attribute='content') | first | default('') -%}\n"
            "{%- set query_text = messages | selectattr('role', 'eq', 'query') | map(attribute='content') | first -%}\n"
            "{%- set document_text = messages | selectattr('role', 'eq', 'document') | map(attribute='content') | first -%}\n"
            "{{- '[CLS]' -}}\n"
            "{%- if instruction -%}{{ instruction }} {%- endif -%}\n"
            "{{- '[Q] ' + query_text + '[SEP][D] ' + document_text + '[SEP]' -}}"
        )
        (save_directory / "chat_template.jinja").write_text(
            chat_template, encoding="utf-8"
        )

        # 5. Update tokenizer_config.json with chat_template if present
        tok_cfg_path = save_directory / "tokenizer_config.json"
        if tok_cfg_path.is_file():
            try:
                tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
                tok_cfg["chat_template"] = chat_template
                tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning(
                    f"Failed to update tokenizer_config.json with chat_template: {e}"
                )

    def export_to_st(
        self,
        save_directory: str,
        repo_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        tb_path: Optional[Union[str, Path]] = None,
        retriever: Optional[str] = None,
        k: Optional[Union[int, str]] = None,
        training_strat: Optional[str] = None,
    ):
        """Export model in Sentence Transformers compatible format, including model card."""
        from xpm_torch.module import assemble_readme_sections

        save_path = Path(save_directory)
        self._last_export_dir = save_path
        if repo_id:
            self.model_name = repo_id

        current_metadata = dict(getattr(self, "_export_metadata", {}) or {})
        if metadata:
            current_metadata.update(metadata)
        if retriever:
            current_metadata["retriever"] = retriever
        if k:
            current_metadata["k"] = k
        if training_strat:
            current_metadata["training_strat"] = training_strat
            current_metadata.setdefault("loss", training_strat)
        if tb_path:
            current_metadata["tb_path"] = str(tb_path)
            self._tb_path = Path(tb_path)

        self._export_metadata = current_metadata

        self.write_hub_extras(save_path, tb_path=tb_path)
        self._write_st_metadata(save_path)
        sections = self.hub_readme_sections()
        if sections:
            readme = assemble_readme_sections([], sections)
            (save_path / "README.md").write_text(readme)

        logger.info(f"Model exported for Sentence Transformers at {save_path}")

    def save_model(self, path: Path):
        """Save the model architecture config, state dict weights, and tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Disconnect base model remote reference to prevent HF Hub network requests on reload
        model_name_id = getattr(self, "hf_id", path.name)
        self.hf_model.config._name_or_path = model_name_id
        if hasattr(self.hf_model.config, "base_model_name_or_path"):
            self.hf_model.config.base_model_name_or_path = model_name_id

        # Set default ST activation function in config to Identity (raw logits)
        self.hf_model.config.sbert_ce_default_activation_function = (
            "torch.nn.modules.linear.Identity"
        )
        self.hf_model.config.sentence_transformers = {
            "activation_fn": "torch.nn.modules.linear.Identity"
        }

        # 1. Save HF model (config.json + model.safetensors file)
        self.hf_model.save_pretrained(path, safe_serialization=True)
        logger.info(f"Saved ModernBERT MICE model configuration and weights to {path}")

        # 2. Save tokenizer
        if hasattr(self.tokenizer, "save_pretrained"):
            if hasattr(self.tokenizer, "name_or_path"):
                self.tokenizer.name_or_path = str(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Saved tokenizer to {path}")
        elif (
            hasattr(self.tokenizer, "tokenizer")
            and self.tokenizer.tokenizer is not None
        ):
            self.tokenizer.tokenizer.name_or_path = str(path)
            self.tokenizer.tokenizer.save_pretrained(path)
            logger.info(f"Saved tokenizer to {path}")
        else:
            logger.warning(
                f"Could not save tokenizer to {path}: no `save_pretrained` method found on tokenizer {type(self.tokenizer)}"
            )

    def load_model(self, path: Path, ignore_top: bool = False):
        """Load model architecture config, state dict weights, and tokenizer from directory."""
        path = Path(path)
        if path.is_file():
            # Always try to load from the parent directory if a file is provided
            path = path.parent

        if not path.exists():
            raise FileNotFoundError(f"Model directory '{path}' does not exist.")

        if not self._initialized:
            if (path / "config.json").exists():
                logger.info(f"Loading MICE config from {path / 'config.json'}")
                mice_config = AutoConfig.from_pretrained(path)
                self.apply_config_overrides(mice_config, source_path_or_id=path)

                hf_model = ModernBertMiceForSequenceClassification(mice_config)
                self.add_module("hf_model", hf_model)
                self.eval_sparsity_tracker = None
                self._initialized = True
            else:
                self.initialize()

            if hasattr(self, "fabric") and self.fabric is not None:
                self.hf_model.to(self.fabric.device)
        else:
            if (path / "config.json").exists():
                logger.info(
                    f"Loading checkpoint weights into initialized MICE model from {path} (device: {self.device})"
                )
                mice_config = AutoConfig.from_pretrained(path)
                self.apply_config_overrides(mice_config, source_path_or_id=path)

        # Ensure tokenizer is initialized in the main process before DataLoader workers fork
        if (path / "tokenizer_config.json").exists() or (
            path / "tokenizer.json"
        ).exists():
            self.tokenizer.model_id = str(path)
        if not self.tokenizer._initialized:
            self.tokenizer.initialize()

        if (path / "model" / "model.safetensors").exists() or (
            path / "model" / "pytorch_model.bin"
        ).exists():
            path = path / "model"

        def _load_and_validate(sd, filename):
            for old_k, new_k in [
                ("classifier.weight", "output_modules.dense_score.linear.weight"),
                ("classifier.bias", "output_modules.dense_score.linear.bias"),
                ("top_final_norm.weight", "output_modules.norm.norm.weight"),
                ("top_final_norm.bias", "output_modules.norm.norm.bias"),
            ]:
                if old_k in sd and new_k not in sd:
                    sd[new_k] = sd.pop(old_k)

            top_prefixes = (
                "top_layers.",
                "output_modules.",
                "classifier.",
                "adapter.",
            )
            if ignore_top:
                logger.info(
                    "[load_model] ignore_top=True: Discarding checkpoint top interaction layers and head weights."
                )
                for k in list(sd.keys()):
                    if k.startswith(top_prefixes):
                        sd.pop(k)

            # PyTorch load_state_dict will natively raise RuntimeError on any size mismatch
            load_res = self.hf_model.load_state_dict(sd, strict=False)
            if load_res.unexpected_keys:
                logger.warning(
                    "[load_model] Unexpected keys in state_dict from %s: %s",
                    filename,
                    load_res.unexpected_keys,
                )
            if load_res.missing_keys:
                ignored_prefixes = [
                    "utility_predictor",
                    "doc_compress",
                    "doc_decompress",
                ]
                if ignore_top:
                    ignored_prefixes.extend(
                        ["top_layers", "output_modules", "classifier", "adapter"]
                    )

                critical_missing = [
                    k
                    for k in load_res.missing_keys
                    if not any(k.startswith(prefix) for prefix in ignored_prefixes)
                ]
                if critical_missing:
                    raise RuntimeError(
                        f"Critical weights missing when loading from {filename}: {critical_missing}"
                    )
                logger.debug(
                    "[load_model] Non-critical keys missing when loading from %s (newly initialized): %s",
                    filename,
                    load_res.missing_keys,
                )
            logger.info("Loaded weights from %s", path / filename)

        if (path / "model.safetensors").exists():
            from safetensors.torch import load_file

            state_dict = load_file(path / "model.safetensors")
            _load_and_validate(state_dict, "model.safetensors")
        elif (path / "pytorch_model.bin").exists():
            state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu")
            _load_and_validate(state_dict, "pytorch_model.bin")
        else:
            super().load_model(path)

    def loader_config(self, path: Path, *, settings=None) -> "SimpleModuleLoader":
        return SimpleModuleLoader.C(value=self, path=path, settings=settings)


class MiceV2DocumentEncoder(TextEncoderBase):
    """Document encoder for ModernBERT MICE v2, interfacing with PlaidIndexBuilder."""

    model: Param[ModernBertMiceScorer]
    max_doc_length: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Maximum number of tokens to store per document in PLAID index. If None, defaults to model's tokenizer max_doc_length."""

    def __initialize__(self) -> None:
        super().__initialize__()
        self.model.initialize()

    @property
    def dimension(self):
        doc_bot_dim = getattr(self.model.hf_model.config, "doc_bottleneck_dim", None)
        if doc_bot_dim is not None and doc_bot_dim > 0:
            return doc_bot_dim
        top_config = getattr(
            self.model.hf_model, "top_config", self.model.hf_model.config
        )
        return getattr(top_config, "hidden_size", 768)

    @staticmethod
    def _token_mask(output: TokensRepresentationOutput) -> Optional[torch.Tensor]:
        mask = output.tokenized.mask
        if mask is None:
            return None
        return mask.to(output.value.device).bool()

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        """Encode a batch of documents and return per-token embeddings filtering padding out."""
        output = self.encode_documents(records)
        mask = self._token_mask(output)
        value = output.value
        if mask is None:
            return [value[i] for i in range(value.shape[0])]
        return [value[i][mask[i]] for i in range(value.shape[0])]

    def encode_documents(
        self, records: List[IDTextRecord]
    ) -> TokensRepresentationOutput:
        effective_max_doc_len = (
            self.max_doc_length
            if self.max_doc_length is not None
            else self.model.tokenizer.max_doc_length
        )
        options = TokenizerOptions(max_length=effective_max_doc_len)
        output = self(records, options=options)
        return output

    def forward(
        self,
        inputs: List[IDTextRecord],
        *args,
        options: Optional[TokenizerOptions] = None,
    ) -> EncoderOutput:
        assert len(args) == 0, "Unhandled extra arguments"
        tokenized = self.model.batch_tokenize(
            inputs, options=options, input_nature=QueryDocInput.DOCUMENT
        )
        if tokenized.ids.device != self.model.device:
            tokenized = tokenized.to(self.model.device)
        encoded, doc_mask = self.model.encode_documents(tokenized.ids, tokenized.mask)
        # Hard pruning shortens encoded's sequence dimension, so the mask returned
        # downstream must reflect the post-pruning mask, not the original tokenizer mask.
        # NOTE: can't use `tokenized._replace(mask=...)` here -- TokenizedTexts overrides
        # __len__ to return len(self.ids) (the batch size) instead of the field count,
        # which breaks NamedTuple._make()'s internal `len(result) != num_fields` check.
        tokenized = TokenizedTexts(**{**tokenized._asdict(), "mask": doc_mask})
        return TokensEncoderOutput(tokenized, encoded)


class MiceV2QueryEncoder(AbstractModuleScorer):
    """Query encoder adapter for ModernBertMiceScorer with custom max_query_length for PLAID search."""

    model: Param[ModernBertMiceScorer]
    max_query_length: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Maximum number of query tokens for PLAID search. If None, defaults to model's tokenizer max_query_length."""

    def __initialize__(self) -> None:
        super().__initialize__()
        self.model.initialize()

    @property
    def device(self):
        return self.model.device

    def query_token_embeddings(self, records: List[IDTextRecord]) -> torch.Tensor:
        return self.model.query_token_embeddings(
            records, max_query_length=self.max_query_length
        )


def mice_v2_scorer(
    hf_id: str = "jhu-clsp/ettin-encoder-17m",
    n_contextualization_layers: int = 4,
    n_docs_ctx_layers: Optional[int] = None,
    n_interaction_layers: Optional[int] = None,
    bound_bottom_layers: bool = True,
    attn_implementation: Optional[str] = None,
    utility_pred: bool = False,
    hard_pruning: Optional[bool] = None,
    hard_prune_threshold: float = 0.5,
    normalize_doc_embeds: bool = False,
    use_upper_rope: bool = True,
    top_layers_doc_ln: bool = False,
    bottom_final_norm: bool = True,
    top_final_norm: bool = False,
    utility_temp: float = 1.0,
    utility_bias: float = 5.0,
    gate_target: str = "h",
    use_hard_concrete: bool = False,
    hc_gamma: float = -0.1,
    hc_zeta: float = 1.1,
    compress_dim: float = 1.0,
    compress_init_option: str = "seed",
    doc_bottleneck_dim: Optional[int] = None,
    use_cross_attn: bool = True,
    head_type: str = "linear",
    max_query_length: Optional[int] = None,
    max_doc_length: Optional[int] = None,
    max_length: int = 512,
    query_prompt: Optional[str] = None,
    doc_prompt: Optional[str] = None,
    freeze_bottom_layers: bool = False,
) -> Tuple[ModernBertMiceScorer, List[LightweightTask]]:
    """
    Factory function for ModernBERT MICE V2 scorer.

    `compress_init_option="seed"` has fallback logic (seed from the pretrained
    checkpoint when compress_dim == 1.0, else fall back to "random") that's
    resolved here rather than later inside `InitMICEV2ModernBERTFromHFID.execute()`,
    so the *resolved* value is what actually lands in the returned
    `ModernBertMiceScorer`'s Param and thus its experimaestro identifier. This
    means e.g. `compress_dim=2.0, compress_init_option="seed"` and
    `compress_dim=2.0, compress_init_option="random"` -- which are functionally
    identical, since "seed" always falls back to "random" once compress_dim !=
    1.0 -- build the *same* scorer config and hash to the same task, so a grid
    search sweeping both doesn't waste compute training the same model twice.
    """
    if compress_init_option == "seed" and compress_dim != 1.0:
        compress_init_option = "random"

    if hard_pruning is None:
        hard_pruning = True if use_hard_concrete else False

    tokenizer = MICEQueryDocTokenizer.C(
        model_id=hf_id,
        max_query_length=max_query_length,
        max_doc_length=max_doc_length,
        max_length=max_length,
        query_prompt=query_prompt,
        doc_prompt=doc_prompt,
    )

    scorer = ModernBertMiceScorer.C(
        hf_id=hf_id,
        tokenizer=tokenizer,
        n_contextualization_layers=n_contextualization_layers,
        n_docs_ctx_layers=n_docs_ctx_layers,
        n_interaction_layers=n_interaction_layers,
        bound_bottom_layers=bound_bottom_layers,
        attn_implementation=attn_implementation,
        utility_pred=utility_pred,
        hard_pruning=hard_pruning,
        hard_prune_threshold=hard_prune_threshold,
        normalize_doc_embeds=normalize_doc_embeds,
        use_upper_rope=use_upper_rope,
        top_layers_doc_ln=top_layers_doc_ln,
        bottom_final_norm=bottom_final_norm,
        top_final_norm=top_final_norm,
        utility_temp=utility_temp,
        utility_bias=utility_bias,
        gate_target=gate_target,
        use_hard_concrete=use_hard_concrete,
        hc_gamma=hc_gamma,
        hc_zeta=hc_zeta,
        compress_dim=compress_dim,
        compress_init_option=compress_init_option,
        doc_bottleneck_dim=doc_bottleneck_dim,
        use_cross_attn=use_cross_attn,
        head_type=head_type,
    )

    init_task = InitMICEV2ModernBERTFromHFID.C(
        scorer=scorer,
        freeze_bottom_layers=freeze_bottom_layers,
    )

    return scorer, [init_task]
