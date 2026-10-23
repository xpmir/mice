"""
Unit tests for MicePlaidRetrieverFactory launcher routing during indexation.
Run with: `uv run pytest tests/test_mice_plaid_launcher_routing.py -v`
"""

from unittest.mock import MagicMock, patch
import pytest

from configuration import Indexation, PlaidConfiguration
from retrievers import MicePlaidRetrieverFactory


class DummyLauncher:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Launcher({self.name})"


class DummyDocuments:
    def __init__(self, doc_id: str):
        self.id = doc_id


class DummyCollection:
    def __init__(self, doc_id: str):
        self.dataset = MagicMock()
        self.dataset.documents = DummyDocuments(doc_id)


class DummyEvaluationsCollection:
    def __init__(self, collections: dict, long_evals: list = None):
        self.collection = collections
        self.long_evals = long_evals if long_evals is not None else ["fever", "msmarco"]


def test_indexation_configuration_defaults():
    """Verify that Indexation configuration has long_evals=None and long_requirements=None by default."""
    cfg = Indexation()
    assert cfg.long_evals is None
    assert cfg.long_requirements is None


def test_mice_plaid_routes_to_long_launcher_index_using_evaluations_long_evals():
    """Verify that build_indexes routes matching datasets to long_launcher_index
    using evaluation's long_evals by default."""
    launcher_default = DummyLauncher("default_index")
    launcher_long = DummyLauncher("long_index")

    collections = {
        "beir/scifact": DummyCollection("scifact_docs"),
        "beir/fever": DummyCollection("fever_docs"),
        "beir/msmarco": DummyCollection("msmarco_docs"),
    }
    evaluations = DummyEvaluationsCollection(collections, long_evals=["fever", "msmarco"])

    mock_model = MagicMock()
    mock_model.get_document_encoder.return_value = MagicMock()

    factory = MicePlaidRetrieverFactory(
        mice_model=mock_model,
        first_stage_factory=MagicMock(),
        topk=100,
        batchsize=32,
    )

    submitted_launchers = {}

    def fake_submit(launcher, init_tasks=None):
        return launcher

    with patch("retrievers.PlaidIndexBuilder.C") as mock_builder_cls:
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance
        mock_builder_instance.submit.side_effect = lambda launcher, **kwargs: launcher

        factory.build_indexes(
            evaluations=evaluations,
            cfg=PlaidConfiguration(),
            batch_size=32,
            seed=0,
            launcher_index=launcher_default,
            init_tasks=[],
            long_launcher_index=launcher_long,
            long_evals=None,  # Should default to evaluations.long_evals
        )

        calls = mock_builder_instance.submit.call_args_list
        assert len(calls) == 3

        # Scifact should route to default index launcher
        assert calls[0].kwargs["launcher"] == launcher_default
        # Fever should route to long index launcher
        assert calls[1].kwargs["launcher"] == launcher_long
        # MSMARCO should route to long index launcher
        assert calls[2].kwargs["launcher"] == launcher_long


def test_mice_plaid_no_fallback_when_long_launcher_index_is_none():
    """Verify that when long_launcher_index is None, all datasets route to launcher_index."""
    launcher_default = DummyLauncher("default_index")

    collections = {
        "beir/scifact": DummyCollection("scifact_docs"),
        "beir/fever": DummyCollection("fever_docs"),
    }
    # Even if evaluations has a long_evals trigger, long_launcher_index is None
    evaluations = DummyEvaluationsCollection(collections, long_evals=["fever"])

    mock_model = MagicMock()
    mock_model.get_document_encoder.return_value = MagicMock()

    factory = MicePlaidRetrieverFactory(
        mice_model=mock_model,
        first_stage_factory=MagicMock(),
        topk=100,
        batchsize=32,
    )

    with patch("retrievers.PlaidIndexBuilder.C") as mock_builder_cls:
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance

        factory.build_indexes(
            evaluations=evaluations,
            cfg=PlaidConfiguration(),
            batch_size=32,
            seed=0,
            launcher_index=launcher_default,
            init_tasks=[],
            long_launcher_index=None,  # No long launcher configured!
        )

        calls = mock_builder_instance.submit.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["launcher"] == launcher_default
        assert calls[1].kwargs["launcher"] == launcher_default


def test_mice_plaid_explicit_long_evals_override():
    """Verify that explicit long_evals passed to build_indexes overrides evaluations.long_evals."""
    launcher_default = DummyLauncher("default_index")
    launcher_long = DummyLauncher("long_index")

    collections = {
        "beir/scifact": DummyCollection("scifact_docs"),
        "beir/fever": DummyCollection("fever_docs"),
    }
    # evaluations says fever is long, but explicit long_evals says only scifact is long
    evaluations = DummyEvaluationsCollection(collections, long_evals=["fever"])

    mock_model = MagicMock()
    mock_model.get_document_encoder.return_value = MagicMock()

    factory = MicePlaidRetrieverFactory(
        mice_model=mock_model,
        first_stage_factory=MagicMock(),
        topk=100,
        batchsize=32,
    )

    with patch("retrievers.PlaidIndexBuilder.C") as mock_builder_cls:
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance

        factory.build_indexes(
            evaluations=evaluations,
            cfg=PlaidConfiguration(),
            batch_size=32,
            seed=0,
            launcher_index=launcher_default,
            init_tasks=[],
            long_launcher_index=launcher_long,
            long_evals=["scifact"],  # Explicit override
        )

        calls = mock_builder_instance.submit.call_args_list
        assert len(calls) == 2
        # scifact routes to long
        assert calls[0].kwargs["launcher"] == launcher_long
        # fever routes to default
        assert calls[1].kwargs["launcher"] == launcher_default


def test_mice_plaid_enforces_compress_only_true_by_default():
    """Verify that MicePlaidRetrieverFactory enforces compress_only=True even if cfg defaults to False."""
    collections = {"beir/scifact": DummyCollection("scifact_docs")}
    evaluations = DummyEvaluationsCollection(collections)

    mock_model = MagicMock()
    mock_model.get_document_encoder.return_value = MagicMock()

    factory = MicePlaidRetrieverFactory(
        mice_model=mock_model,
        first_stage_factory=MagicMock(),
        topk=100,
        batchsize=32,
    )

    with patch("retrievers.PlaidIndexBuilder.C") as mock_builder_cls:
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance

        # Default PlaidConfiguration has compress_only=False
        cfg = PlaidConfiguration()
        assert cfg.compress_only is False

        factory.build_indexes(
            evaluations=evaluations,
            cfg=cfg,
            batch_size=32,
            seed=0,
            launcher_index=DummyLauncher("default"),
            init_tasks=[],
        )

        assert mock_builder_cls.call_count == 1
        assert mock_builder_cls.call_args.kwargs["compress_only"] is True


def test_mice_plaid_compress_only_false_when_use_colbert_first_stage():
    """Verify that when use_colbert_first_stage=True, compress_only is set to False for IVF retrieval."""
    collections = {"beir/scifact": DummyCollection("scifact_docs")}
    evaluations = DummyEvaluationsCollection(collections)

    mock_model = MagicMock()
    mock_model.get_document_encoder.return_value = MagicMock()

    factory = MicePlaidRetrieverFactory(
        mice_model=mock_model,
        first_stage_factory=MagicMock(),
        topk=100,
        batchsize=32,
    )

    with patch("retrievers.PlaidIndexBuilder.C") as mock_builder_cls:
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance

        cfg = PlaidConfiguration(use_colbert_first_stage=True)

        factory.build_indexes(
            evaluations=evaluations,
            cfg=cfg,
            batch_size=32,
            seed=0,
            launcher_index=DummyLauncher("default"),
            init_tasks=[],
        )

        assert mock_builder_cls.call_count == 1
        assert mock_builder_cls.call_args.kwargs["compress_only"] is False


def test_mice_plaid_warns_and_overrides_when_colbert_and_compress_only_true(caplog):
    """Verify that if use_colbert_first_stage=True and compress_only=True, it warns and overrides to False."""
    collections = {"beir/scifact": DummyCollection("scifact_docs")}
    evaluations = DummyEvaluationsCollection(collections)

    mock_model = MagicMock()
    mock_model.get_document_encoder.return_value = MagicMock()

    factory = MicePlaidRetrieverFactory(
        mice_model=mock_model,
        first_stage_factory=MagicMock(),
        topk=100,
        batchsize=32,
    )

    with patch("retrievers.PlaidIndexBuilder.C") as mock_builder_cls:
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance

        cfg = PlaidConfiguration(use_colbert_first_stage=True, compress_only=True)

        import logging
        with caplog.at_level(logging.WARNING):
            factory.build_indexes(
                evaluations=evaluations,
                cfg=cfg,
                batch_size=32,
                seed=0,
                launcher_index=DummyLauncher("default"),
                init_tasks=[],
            )

        assert mock_builder_cls.call_count == 1
        assert mock_builder_cls.call_args.kwargs["compress_only"] is False
        assert any("use_colbert_first_stage is enabled but compress_only is True" in r.message for r in caplog.records)



def test_plaid_retriever_routes_to_long_launcher_index_for_hotpotqa():
    """Verify that plaid_retriever with key='hotpotqa_decontaminated' routes to long_launcher_index."""
    from retrievers import plaid_retriever
    import index_utils

    launcher_default = DummyLauncher("default_index")
    launcher_long = DummyLauncher("long_index")

    mock_cfg = MagicMock()
    mock_cfg.plaid = PlaidConfiguration()
    mock_cfg.retrieval.k = 100
    mock_cfg.indexation = Indexation()

    docs = DummyDocuments("ai.lighton.beir_decontaminated.hotpotqa.documents")

    with (
        patch("index_utils._indexes", {}),
        patch("index_utils.PlaidIndexBuilder.C") as mock_builder_cls,
        patch("retrievers.PlaidRetriever.C") as mock_retriever_cls,
    ):
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance
        mock_builder_cls.return_value.__identifier__ = MagicMock(return_value="plaid_mock_id_hotpot")

        # Call plaid_retriever with hotpotqa_decontaminated key and trigger ["hotpotqa"]
        plaid_retriever(
            cfg=mock_cfg,
            name="LateOn",
            encoder=MagicMock(),
            documents=docs,
            launcher_index=launcher_default,
            long_launcher_index=launcher_long,
            long_evals=["fever", "dpbedia", "hotpotqa"],
            key="hotpotqa_decontaminated",
        )

        calls = mock_builder_instance.submit.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["launcher"] == launcher_long


def test_plaid_retriever_routes_to_default_for_non_matching_key():
    """Verify that plaid_retriever with non-matching key routes to launcher_index."""
    from retrievers import plaid_retriever
    import index_utils

    launcher_default = DummyLauncher("default_index")
    launcher_long = DummyLauncher("long_index")

    mock_cfg = MagicMock()
    mock_cfg.plaid = PlaidConfiguration()
    mock_cfg.retrieval.k = 100
    mock_cfg.indexation = Indexation()

    docs = DummyDocuments("beir.scifact.documents")

    with (
        patch("index_utils._indexes", {}),
        patch("index_utils.PlaidIndexBuilder.C") as mock_builder_cls,
        patch("retrievers.PlaidRetriever.C") as mock_retriever_cls,
    ):
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance
        mock_builder_cls.return_value.__identifier__ = MagicMock(return_value="plaid_mock_id_scifact")

        plaid_retriever(
            cfg=mock_cfg,
            name="LateOn",
            encoder=MagicMock(),
            documents=docs,
            launcher_index=launcher_default,
            long_launcher_index=launcher_long,
            long_evals=["fever", "dpbedia", "hotpotqa"],
            key="scifact",
        )

        calls = mock_builder_instance.submit.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["launcher"] == launcher_default


def test_get_splade_index_routes_to_long_launcher_index():
    """Verify that get_splade_index routes matching dataset to long_launcher_index."""
    from index_utils import get_splade_index
    import index_utils

    launcher_default = DummyLauncher("default_index")
    launcher_long = DummyLauncher("long_index")

    indexation_cfg = Indexation(batch_size=128)
    docs = DummyDocuments("beir.hotpotqa.documents")
    docs.count = 1000

    with patch("index_utils._indexes", {}), patch("index_utils.SparseRetrieverIndexBuilder.C") as mock_builder_cls:
        mock_builder_instance = MagicMock()
        mock_builder_cls.return_value.tag.return_value = mock_builder_instance
        mock_builder_cls.return_value.__identifier__ = MagicMock(return_value="splade_mock_id_hotpot")

        get_splade_index(
            documents=docs,
            splade_encoder=MagicMock(),
            indexation_cfg=indexation_cfg,
            launcher_index=launcher_default,
            long_launcher_index=launcher_long,
            long_evals=["hotpotqa"],
            key="hotpotqa_decontaminated",
        )

        calls = mock_builder_instance.submit.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["launcher"] == launcher_long


def test_mice_plaid_retriever_factory_call_accepts_splade_factory_with_documents_param():
    """Verify that MicePlaidRetrieverFactory.__call__ passes dataset positionally so factories
    expecting 'documents' (like splade_retriever partial) succeed without TypeError."""
    mock_first_stage = MagicMock()

    def splade_factory(documents, key=None):
        mock_first_stage(documents, key=key)
        return MagicMock()

    mock_model = MagicMock()
    factory = MicePlaidRetrieverFactory(
        mice_model=mock_model,
        first_stage_factory=splade_factory,
        topk=100,
        batchsize=32,
    )
    docs = DummyDocuments("scifact_docs")
    factory.indexes = {"beir/scifact": MagicMock()}
    factory.documents = {"beir/scifact": docs}

    with patch("MICE.plaid_mice.MicePlaidRetrieverv2.C") as mock_mice_retriever:
        mock_mice_retriever.return_value.tag.return_value = MagicMock()
        factory(docs, key="beir/scifact")
        assert mock_first_stage.call_count == 1
        assert mock_first_stage.call_args[0][0] is docs
