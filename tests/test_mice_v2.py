"""
Tests for ModernBERT MICE V2 Architecture.
Run with: `uv run pytest tests/test_mice_v2.py -v`
"""

import tempfile
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder

from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer, ModernBertMiceScorer
from xpmir.letor.records import PointwiseItems
from experimaestro import LightweightTask, Param

import logging
logging.basicConfig(level=logging.INFO)

class TestMiceV2ForwardTask(LightweightTask):
    """Tests Mice V2 Loading, scoring a PointWiseItem, and reload persistence"""

    scorer: Param[ModernBertMiceScorer]

    def execute(self):
        self.scorer.initialize()
        logging.info(f"Initialized MICE V2 scorer: {self.scorer}")
        queries = ["What is the capital of France?", "Where is Paris located?"]
        documents = [
            "Paris is the capital and most populous city of France.",
            "Paris is located in northern central France on the Seine river.",
        ]
        input_records = PointwiseItems.from_texts(topics=queries, documents=documents)

        self.scorer.eval()
        with torch.no_grad():
            output = self.scorer(input_records)

        assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
        assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"


@pytest.mark.parametrize("model_id", ["jhu-clsp/ettin-encoder-17m"])
@pytest.mark.parametrize(
    "n_contextualization_layers, n_docs_ctx_layers, bound_bottom_layers",
    [
        (2, None, True),
        (2, 3, False),
    ],
)
def test_mice_v2(
    model_id,
    n_contextualization_layers,
    n_docs_ctx_layers,
    bound_bottom_layers,
):
    """Tests MICE V2 forward pass, weight & tokenizer persistence, and standalone ST/HF rebuild."""

    logging.info(f"Testing MICE V2 with model_id={model_id}, n_contextualization_layers={n_contextualization_layers}, n_docs_ctx_layers={n_docs_ctx_layers}, bound_bottom_layers={bound_bottom_layers}")

    logging.info("\n\n#1 - Initializing MICE V2 scorer and tasks...")
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=n_contextualization_layers,
        n_docs_ctx_layers=n_docs_ctx_layers,
        bound_bottom_layers=bound_bottom_layers,
    )

    test_forward_task = TestMiceV2ForwardTask.C(scorer=scorer_cfg).instance()

    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()

    scorer = test_forward_task.scorer

    # --- Persistence Verification ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        save_path = tmp_path / "model"
        st_save_path = tmp_path / "st_model"

        # 1. Save model and tokenizer
        scorer.save_model(save_path)

        assert (save_path / "config.json").exists(), "config.json missing from save_path!"
        assert (save_path / "model.safetensors").exists() or (save_path / "pytorch_model.bin").exists(), "Weights missing from save_path!"
        assert (save_path / "tokenizer_config.json").exists(), "tokenizer_config.json missing from save_path!"

        # 2. Verify standalone rebuild without base model using HF Auto Classes
        standalone_tok = AutoTokenizer.from_pretrained(save_path)
        assert standalone_tok is not None, "Failed to load tokenizer from save_path!"

        standalone_model = AutoModelForSequenceClassification.from_pretrained(save_path)
        assert standalone_model is not None, "Failed to rebuild model from save_path without base model!"

        # 3. Reload ModernBertMiceScorer
        logging.info("\n\n#3 - Reloading ModernBertMiceScorer...")
        reloaded_scorer = ModernBertMiceScorer.C(
            hf_id=model_id,
            tokenizer=scorer_cfg.tokenizer,
            n_contextualization_layers=n_contextualization_layers,
        ).instance()
        reloaded_scorer.load_model(save_path)

        assert torch.allclose(
            scorer.hf_model.classifier.weight, reloaded_scorer.hf_model.classifier.weight
        ), "Classifier weights mismatch after standard reload!"

        # 4. SentenceTransformers Export & Import Roundtrip
        scorer.export_to_st(str(st_save_path))

        logging.info("\n\n#4 - Loading exported model with SentenceTransformers CrossEncoder...")
        st_cross_encoder = CrossEncoder(str(st_save_path))
        scores = st_cross_encoder.predict([["What is the capital of France?", "Paris is the capital of France."]])
        assert scores is not None
        assert len(scores) == 1


@pytest.mark.parametrize("use_upper_rope", [True, False])
def test_mice_v2_use_upper_rope(use_upper_rope):
    """Tests MICE V2 forward pass and serialization with use_upper_rope enabled vs disabled."""
    model_id = "jhu-clsp/ettin-encoder-17m"
    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=model_id,
        n_contextualization_layers=2,
        use_upper_rope=use_upper_rope,
    )

    test_forward_task = TestMiceV2ForwardTask.C(scorer=scorer_cfg).instance()
    for init_task in init_tasks:
        init_task.instance().execute()

    test_forward_task.execute()
    scorer = test_forward_task.scorer

    assert scorer.hf_model.config.use_upper_rope == use_upper_rope

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model"
        scorer.save_model(save_path)

        standalone_model = AutoModelForSequenceClassification.from_pretrained(save_path)
        assert standalone_model.config.use_upper_rope == use_upper_rope


def test_export_model_learner_tag_matching():
    """Tests export_model Learner lookup filtering out evaluation-only tags (first_stage, checkpoint)."""
    from training_utils import export_model
    import pandas as pd
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        job_dir = tmp_path / "job"
        job_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = tmp_path / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        (tb_dir / "events.out.tfevents.123").write_text("tensorboard trace")

        mock_learner = MagicMock()
        mock_learner.jobpath = job_dir
        mock_learner.logpath = tb_dir
        xpm_learner = MagicMock()
        xpm_learner.tags.return_value = {"base": "ettin-17m", "seed": "42"}
        setattr(mock_learner, "__xpm__", xpm_learner)

        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_loader.model = mock_model
        xpm_loader = MagicMock()
        xpm_loader.tags.return_value = {
            "base": "ettin-17m",
            "seed": "42",
            "first_stage": "bm25",
            "checkpoint": "last",
        }
        setattr(mock_loader, "__xpm__", xpm_loader)

        mock_action_inst = MagicMock()
        mock_loader.export_action.return_value.instance.return_value = mock_action_inst

        best_tags = {
            "base": "ettin-17m",
            "seed": "42",
            "first_stage": "bm25",
            "checkpoint": "last",
        }

        csv_df = pd.DataFrame({"dataset": ["msmarco"], "nDCG@10": [0.38]})
        md_df = pd.DataFrame({"dataset": ["msmarco"], "nDCG@10": [0.38]})

        mock_cfg = MagicMock()
        mock_cfg.learner.loss = "marginMSE"
        mock_cfg.retrieval.k = 1000

        export_model(
            best_tags=best_tags,
            model_name="TestModel",
            csv_results=csv_df,
            md_results=md_df,
            learners=[mock_learner],
            all_weights=[mock_loader],
            best_cfg=mock_cfg,
            resultspath=tmp_path,
        )

        model_dir = tmp_path / "models" / "TestModel"
        assert mock_action_inst.loader.model._tb_path == tb_dir
        assert mock_action_inst.loader.model._export_metadata["tb_path"] == str(tb_dir)
        assert mock_action_inst.loader.model._export_metadata["training_strat"] == "MarginMSE"


def test_export_model_removes_stale_readme():
    """Verify that export_model removes an existing README.md before creating a new export."""
    from training_utils import export_model
    import pandas as pd
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_dir = tmp_path / "models" / "TestModel"
        model_dir.mkdir(parents=True, exist_ok=True)
        stale_readme = model_dir / "README.md"
        stale_readme.write_text("old stale readme")
        assert stale_readme.exists()

        mock_loader = MagicMock()
        mock_loader.model = MagicMock()
        xpm_loader = MagicMock()
        xpm_loader.tags.return_value = {"base": "ettin-17m"}
        setattr(mock_loader, "__xpm__", xpm_loader)
        mock_action_inst = MagicMock()
        mock_loader.export_action.return_value.instance.return_value = mock_action_inst

        csv_df = pd.DataFrame({"dataset": ["msmarco"], "nDCG@10": [0.42]})
        md_df = pd.DataFrame({"dataset": ["msmarco"], "nDCG@10": [0.42]})

        mock_cfg = MagicMock()
        mock_cfg.learner.loss = "marginMSE"
        mock_cfg.retrieval.k = 1000

        export_model(
            best_tags={"base": "ettin-17m"},
            model_name="TestModel",
            csv_results=csv_df,
            md_results=md_df,
            learners=[],
            all_weights=[mock_loader],
            best_cfg=mock_cfg,
            resultspath=tmp_path,
        )

        # Stale readme was deleted before export_to_folder was called
        assert not stale_readme.exists()


def test_process_experiment_results_export_only_best():
    """Verify that export_only_best=True exports only the winning candidate model."""
    from training_utils import process_experiment_results
    from experimaestro import RunMode
    from configuration import CE_FineTuning
    import pandas as pd
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Mock tests dataframe with 2 LR configs matching experimaestro to_dataframe MultiIndex schema
        columns = pd.MultiIndex.from_tuples([
            ("dataset", ""),
            ("tag", "base"),
            ("tag", "lr"),
            ("tag", "checkpoint"),
            ("tag", "first_stage"),
            ("tag", "seed"),
            ("metric", "nDCG@10"),
        ])
        eval_df = pd.DataFrame(
            [
                ["BEIR13", "ettin-17m", 7e-5, "last", "bm25", 0, 0.5739],
                ["BEIR13", "ettin-17m", 3e-5, "last", "bm25", 0, 0.5760],
            ],
            columns=columns,
        )

        mock_tests = MagicMock()
        mock_tests.to_dataframe.return_value = eval_df
        mock_tests.collection = {}

        mock_helper = MagicMock()
        mock_helper.xp.run_mode = RunMode.NORMAL
        mock_helper.xp.resultspath = tmp_path

        mock_cfg = CE_FineTuning(export_trained_models=True, export_only_best=True)
        all_tags = [{"lr": 7e-5}, {"lr": 3e-5}]
        config_map = {
            frozenset([("lr", 7e-5)]): mock_cfg,
            frozenset([("lr", 3e-5)]): mock_cfg,
        }

        # Mock loaders for weights
        def make_loader(lr):
            loader = MagicMock()
            loader.model = MagicMock()
            xpm = MagicMock()
            xpm.tags.return_value = {
                "base": "ettin-17m",
                "lr": lr,
                "checkpoint": "last",
                "first_stage": "bm25",
                "seed": 0,
            }
            setattr(loader, "__xpm__", xpm)
            return loader

        loader_7e5 = make_loader(7e-5)
        loader_3e5 = make_loader(3e-5)

        exported_names = []
        def mock_name_fn(tags):
            return "MiceV2-TestModel"

        process_experiment_results(
            tests=mock_tests,
            all_tags=all_tags,
            config_map=config_map,
            cfg=mock_cfg,
            helper=mock_helper,
            learners=[],
            all_weights=[loader_7e5, loader_3e5],
            get_name_fn=mock_name_fn,
        )

        # models/ folder should have exactly 1 exported model directory
        models_dir = tmp_path / "models"
        exported_dirs = list(models_dir.iterdir())
        assert len(exported_dirs) == 1
        assert exported_dirs[0].name == "MiceV2-TestModel"

        # The results.csv inside the exported model directory should be from the winning 3e-5 config (0.5760)
        exported_csv = pd.read_csv(exported_dirs[0] / "results.csv")
        beir_row = exported_csv[exported_csv["dataset"].str.contains("BEIR13", na=False)]
        assert not beir_row.empty
        assert beir_row["nDCG@10"].values[0] == pytest.approx(57.60, abs=0.01)

        # best_models_per_scorer_raw_results.csv should contain both candidates
        raw_summary = pd.read_csv(tmp_path / "best_models_per_scorer_raw_results.csv")
        assert len(raw_summary["lr"].unique()) == 2


def test_process_experiment_results_export_all_disambiguates_collision():
    """Verify that export_only_best=False disambiguates colliding model names with grid tags."""
    from training_utils import process_experiment_results
    from experimaestro import RunMode
    from configuration import CE_FineTuning
    import pandas as pd
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        columns = pd.MultiIndex.from_tuples([
            ("dataset", ""),
            ("tag", "base"),
            ("tag", "lr"),
            ("tag", "checkpoint"),
            ("tag", "first_stage"),
            ("tag", "seed"),
            ("metric", "nDCG@10"),
        ])
        eval_df = pd.DataFrame(
            [
                ["BEIR13", "ettin-17m", 7e-5, "last", "bm25", 0, 0.5739],
                ["BEIR13", "ettin-17m", 3e-5, "last", "bm25", 0, 0.5760],
            ],
            columns=columns,
        )

        mock_tests = MagicMock()
        mock_tests.to_dataframe.return_value = eval_df
        mock_tests.collection = {}

        mock_helper = MagicMock()
        mock_helper.xp.run_mode = RunMode.NORMAL
        mock_helper.xp.resultspath = tmp_path

        mock_cfg = CE_FineTuning(export_trained_models=True, export_only_best=False)
        all_tags = [{"lr": 7e-5}, {"lr": 3e-5}]
        config_map = {
            frozenset([("lr", 7e-5)]): mock_cfg,
            frozenset([("lr", 3e-5)]): mock_cfg,
        }

        def make_loader(lr):
            loader = MagicMock()
            loader.model = MagicMock()
            xpm = MagicMock()
            xpm.tags.return_value = {
                "base": "ettin-17m",
                "lr": lr,
                "checkpoint": "last",
                "first_stage": "bm25",
                "seed": 0,
            }
            setattr(loader, "__xpm__", xpm)
            return loader

        process_experiment_results(
            tests=mock_tests,
            all_tags=all_tags,
            config_map=config_map,
            cfg=mock_cfg,
            helper=mock_helper,
            learners=[],
            all_weights=[make_loader(7e-5), make_loader(3e-5)],
            get_name_fn=lambda tags: "MiceV2-TestModel",
        )

        models_dir = tmp_path / "models"
        exported_dirs = sorted([d.name for d in models_dir.iterdir()])
        assert len(exported_dirs) == 2
        # Disambiguated with grid tag suffixes
        assert any("lr-7e-05" in name or "lr-7e-5" in name for name in exported_dirs)
        assert any("lr-3e-05" in name or "lr-3e-5" in name for name in exported_dirs)
