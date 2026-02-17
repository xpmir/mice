import torch
from typing import Tuple
from experimaestro import Param
from datamaestro_text.data.ir import TextItem
from xpmir.distributed import DistributableModel
from xpmir.learning.batchers import Batcher
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import (
    BaseRecords,
    PairwiseRecords,
)
from xpmir.rankers import LearnableScorer
from xpmir.text import TokenizerOptions
from xpmir.text.encoders import TextEncoderBase, TripletTextEncoder
from xpmir.rankers import (
    DuoLearnableScorer,
    DuoTwoStageRetriever,
    Retriever,
)
from xpmir.utils.utils import easylog
from xpm_torch.xpmModel import xpmTorchHubModule

logger = easylog()


class CrossScorer(LearnableScorer, xpmTorchHubModule):
    """Query-Document Representation Classifier

    Based on a query-document representation representation (e.g. BERT [CLS] token).
    AKA Cross-Encoder
    """

    encoder: Param[TextEncoderBase[Tuple[str, str], torch.Tensor]]
    """an encoder for encoding the concatenated query-document tokens which
    doesn't contains the final linear layer"""

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def __initialize__(self, options):
        super().__initialize__(options)
        self.encoder.initialize(options)
        self.classifier = torch.nn.Linear(self.encoder.dimension, 1)

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        # Encode queries and documents
        pairs = self.encoder(
            [
                (tr[TextItem].text, dr[TextItem].text)
                for tr, dr in zip(inputs.topics, inputs.documents)
            ],
            # options=self.tokenizer_options,
        )  # shape (batch_size * dimension)
        return self.classifier(pairs.value).squeeze(1)

    def distribute_models(self, update):
        self.encoder = update(self.encoder)
