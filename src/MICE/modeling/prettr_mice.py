import torch
import logging
from typing import List, Optional, Tuple

from experimaestro import Param, field, Constant
from datamaestro_ir.data.base import IDTextRecord
from xpmir.letor.records import BaseItems
from xpmir.text import TokenizedTexts
from xpmir.text.tokenizers import TokenizerOptions
from xpmir.text.encoders import (
    EncoderOutput,
    TextEncoderBase,
    TokensEncoderOutput,
    TokensRepresentationOutput,
)
from xpmir.neural.huggingface import HFCrossScorer, InitCEFromHFID
from xpm_torch.utils import to_device

logger = logging.getLogger(__name__)


class PreTTRCrossEncoder(HFCrossScorer):
    """
    PreTTR Baseline implementation wrapped in the MICE framework.
    It uses joint tokenization but prevents cross-attention in the early layers
    using a join_mask.
    """

    join_layer: Param[int] = field(default=6)
    """The layer index at which full self-attention begins."""

    prettr_max_query_length: Param[int] = field(default=32)
    """Legacy PreTTR query length parameter; fixed-offset encoding is kept in garage helper."""

    _version: Constant[int] = 1

    def _garage_shifted_position_ids(
        self, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Garage helper for future fixed-offset PreTTR positional ids."""
        BAT, SEQ = token_type_ids.shape
        position_ids = (
            torch.arange(SEQ, dtype=torch.long, device=token_type_ids.device)
            .unsqueeze(0)
            .expand(BAT, SEQ)
            .clone()
        )

        is_doc = token_type_ids == 1
        for b in range(BAT):
            doc_indices = torch.where(is_doc[b])[0]
            if len(doc_indices) > 0:
                doc_start = doc_indices[0]
                num_doc = len(doc_indices)
                position_ids[b, doc_start:] = torch.arange(
                    self.prettr_max_query_length,
                    self.prettr_max_query_length + num_doc,
                    device=token_type_ids.device,
                )
        return position_ids

    def forward(
        self,
        inputs: Optional[BaseItems] = None,
        tokenized: Optional[TokenizedTexts] = None,
        doc_hidden_states: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ):
        if inputs is None and tokenized is None:
            raise ValueError("Either raw inputs or tokenized inputs must be provided.")

        if tokenized is None:
            tokenized = self.batch_tokenize(inputs)

        input_ids = to_device(tokenized.ids, self.device)
        attention_mask = to_device(tokenized.mask, self.device)
        token_type_ids = to_device(tokenized.token_type_ids, self.device)

        BAT, SEQ = input_ids.shape
        model = self.encoder.model
        base_model = getattr(model, model.base_model_prefix)
        dtype = next(model.parameters()).dtype

        if doc_hidden_states is None:
            # 1. Full joint forward through early layers with join_mask
            hidden_states = base_model.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
            )

            if (
                hasattr(base_model, "embeddings_project")
                and base_model.embeddings_project is not None
            ):
                hidden_states = base_model.embeddings_project(hidden_states)

            # Build join_mask
            b_attention_mask = attention_mask.bool()
            ext_attn_mask = b_attention_mask.reshape(
                BAT, 1, SEQ, 1
            ) * b_attention_mask.reshape(BAT, 1, 1, SEQ)
            join_mask = ~ext_attn_mask | (
                token_type_ids.reshape(BAT, 1, SEQ, 1)
                != token_type_ids.reshape(BAT, 1, 1, SEQ)
            )
            join_mask = join_mask.to(dtype=dtype).masked_fill_(
                join_mask, torch.finfo(dtype).min
            )

            extended_attention_mask = base_model.get_extended_attention_mask(
                attention_mask, input_ids.size()
            ).to(dtype=dtype)

            start_layer = 0
        else:
            # 2. Resuming from precomputed document embeddings
            # We assume inputs contains only the queries (concatenated with [SEP] for joint-like structure)
            # Actually, MicePlaidRetrieverv2 passes tokenized_queries (tokenized_q)
            # which for PreTTR should be the query part [CLS] q [SEP]
            # And doc_hidden_states is a list of [doc [SEP]] embeddings
            def repad_batch(
                doc_embeddings: List[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                docs = [d.squeeze(0) if d.dim() == 3 else d for d in doc_embeddings]
                lengths = [doc.shape[0] for doc in docs]
                max_len = max(lengths)
                device = docs[0].device
                B = len(docs)
                padded = torch.zeros(B, max_len, *docs[0].shape[1:], device=device)
                mask = torch.zeros(B, max_len, device=device).long()
                for i, doc in enumerate(docs):
                    seq_len = doc.shape[0]
                    padded[i, :seq_len] = doc
                    mask[i, :seq_len] = 1
                return padded, mask

            if isinstance(doc_hidden_states, list):
                doc_hidden_states, doc_mask = repad_batch(doc_hidden_states)

            # Encode queries up to join_layer
            # tokenized contains only queries here
            q_hidden = self.encode_queries(input_ids, attention_mask)

            # Concatenate
            hidden_states = torch.cat([q_hidden, doc_hidden_states], dim=1)
            new_attention_mask = torch.cat([attention_mask, doc_mask], dim=1)
            # Join layer mask is not needed anymore as we are at join_layer or above
            # but we still need the extended_attention_mask for the remaining layers
            extended_attention_mask = base_model.get_extended_attention_mask(
                new_attention_mask, hidden_states.shape[:2]
            ).to(dtype=dtype)

            start_layer = self.join_layer
            join_mask = None  # Not used

        # Common loop for remaining layers
        for i in range(start_layer, len(base_model.encoder.layer)):
            layer_module = base_model.encoder.layer[i]
            layer_mask = join_mask if i < self.join_layer else extended_attention_mask
            hidden_states = layer_module(hidden_states, attention_mask=layer_mask)

        # Scoring
        if model.config.model_type in ["roberta", "electra"]:
            logits = model.classifier(hidden_states)
        else:
            if hasattr(base_model, "pooler") and base_model.pooler is not None:
                # Pooler might need to be carefully handled if we sliced the sequence
                # but standard BERT pooler just takes CLS token (hidden_states[:, 0])
                pooled_output = base_model.pooler(hidden_states)
            else:
                pooled_output = hidden_states[:, 0]

            if hasattr(model, "dropout"):
                pooled_output = model.dropout(pooled_output)

            logits = model.classifier(pooled_output)

        return logits.squeeze(-1)

    # --- MICE API Compatibility ---

    def query_token_embeddings(self, records: List[IDTextRecord]) -> List[torch.Tensor]:
        # Queries are encoded normally (type 0, pos 0...)
        options = TokenizerOptions(max_length=self.prettr_max_query_length)
        texts = [r.text if hasattr(r, "text") else r["text"] for r in records]
        tokenized = self.tokenizer.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=options.max_length,
            return_tensors="pt",
        )
        output = self.encode_queries(
            tokenized["input_ids"], tokenized["attention_mask"]
        )
        return [output[i] for i in range(output.shape[0])]

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        # For PreTTR, document encoding must match the document part of joint tokenization
        # Standard joint tokenization for BERT is [CLS] query [SEP] doc [SEP]
        # The document part is 'doc [SEP]' (encoded with token_type_id=1).
        max_len = getattr(self.tokenizer, "max_doc_length", 512)

        # We encode WITHOUT [CLS] to match the 'type=1' behavior in joint forward
        texts = [r.text if hasattr(r, "text") else r["text"] for r in records]
        tokenized = self.tokenizer.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # Manually add [SEP] at the end to match joint tokenization [CLS] query [SEP] doc [SEP]
        sep_id = self.tokenizer.tokenizer.sep_token_id
        input_ids = torch.cat(
            [
                tokenized["input_ids"],
                torch.full(
                    (tokenized["input_ids"].shape[0], 1), sep_id, dtype=torch.long
                ),
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                tokenized["attention_mask"],
                torch.ones((tokenized["attention_mask"].shape[0], 1), dtype=torch.long),
            ],
            dim=1,
        )

        output = self.encode_documents(input_ids, attention_mask)
        return [output[i] for i in range(output.shape[0])]

    def encode_queries(self, input_ids, attention_mask):
        input_ids = to_device(input_ids, self.device)
        attention_mask = to_device(attention_mask, self.device)
        token_type_ids = torch.zeros_like(input_ids)

        model = self.encoder.model
        base_model = getattr(model, model.base_model_prefix)

        hidden_states = base_model.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

        ext_mask = base_model.get_extended_attention_mask(
            attention_mask, input_ids.size()
        )

        for i in range(self.join_layer):
            hidden_states = base_model.encoder.layer[i](
                hidden_states, attention_mask=ext_mask
            )[0]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

        return hidden_states

    def encode_documents(self, input_ids, attention_mask):
        input_ids = to_device(input_ids, self.device)
        attention_mask = to_device(attention_mask, self.device)
        token_type_ids = torch.ones_like(input_ids)

        model = self.encoder.model
        base_model = getattr(model, model.base_model_prefix)

        hidden_states = base_model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        ext_mask = base_model.get_extended_attention_mask(
            attention_mask, input_ids.size()
        )

        for i in range(self.join_layer):
            hidden_states = base_model.encoder.layer[i](
                hidden_states, attention_mask=ext_mask
            )[0]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

        return hidden_states

    def get_document_encoder(self) -> TextEncoderBase:
        return PreTTRDocumentEncoder.C(model=self)


class PreTTRDocumentEncoder(TextEncoderBase):
    """Document encoder using PreTTR independent layers"""

    model: Param[PreTTRCrossEncoder]

    def __initialize__(self) -> None:
        super().__initialize__()
        self.model.initialize()

    @property
    def dimension(self):
        return self.model.encoder.model.config.hidden_size

    @staticmethod
    def _token_mask(output: TokensRepresentationOutput) -> Optional[torch.Tensor]:
        mask = output.tokenized.mask
        if mask is None:
            return None
        return mask.to(output.value.device).bool()

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        """Encode a batch of documents and return the list of per-token
        embeddings, one tensor ``(num_tokens, dim)`` per document. Padding
        positions are filtered out.
        """
        output = self.encode_documents(records)
        mask = self._token_mask(output)
        value = output.value
        if mask is None:
            return [value[i] for i in range(value.shape[0])]
        return [value[i][mask[i]] for i in range(value.shape[0])]

    def encode_documents(
        self, records: List[IDTextRecord]
    ) -> TokensRepresentationOutput:
        # Replicate MiceDocumentEncoder logic
        max_len = getattr(self.model.tokenizer, "max_doc_length", 512)
        tokenized = self.model.tokenizer.tokenizer(
            [r.text for r in records],
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
            return_tensors="pt",
        )

        sep_id = self.model.tokenizer.tokenizer.sep_token_id
        input_ids = torch.cat(
            [
                tokenized["input_ids"],
                torch.full(
                    (tokenized["input_ids"].shape[0], 1), sep_id, dtype=torch.long
                ),
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                tokenized["attention_mask"],
                torch.ones((tokenized["attention_mask"].shape[0], 1), dtype=torch.long),
            ],
            dim=1,
        )

        tokenized_obj = TokenizedTexts(
            ids=input_ids,
            mask=attention_mask,
            token_type_ids=torch.ones_like(input_ids),
            lens=attention_mask.sum(dim=1).tolist(),
        )

        return TokensEncoderOutput(
            tokenized_obj,
            self.model.encode_documents(input_ids, attention_mask),
        )

    def forward(
        self,
        inputs: List[IDTextRecord],
        *args,
        options: Optional[TokenizerOptions] = None,
    ) -> EncoderOutput:
        return self.encode_documents(inputs)


def prettr_scorer(
    hf_id: str,
    join_layer: int,
    max_length: int = 512,
    prettr_max_query_length: int = 32,
) -> Tuple[PreTTRCrossEncoder, List[InitCEFromHFID]]:
    """Factory function for PreTTR scorer"""
    from xpmir.neural.huggingface import (
        HFSequenceClassification,
        HFConfigID,
        HFQueryDocTokenizer,
    )

    encoder = HFSequenceClassification.C(config=HFConfigID.C(hf_id=hf_id))
    tokenizer = HFQueryDocTokenizer.C(model_id=hf_id, max_length=max_length)

    scorer = PreTTRCrossEncoder.C(
        encoder=encoder,
        tokenizer=tokenizer,
        join_layer=join_layer,
        prettr_max_query_length=prettr_max_query_length,
    )

    return scorer, [InitCEFromHFID.C(model=encoder)]
