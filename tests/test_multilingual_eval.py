"""Multilingual evaluation and validation sets for mMICE-P (mmicep_brief.md, step 4)."""

from pathlib import Path
from collections import namedtuple
from unittest.mock import MagicMock

import pytest
import yaml

import validations
from configuration import LIGHTON_LANGUAGES, Validation
from tests import MIRACL_KEYS, MLDR_KEYS, NANO_BEIR_KEYS, test_registry

CONFIG_DIR = Path(__file__).parents[1] / "experiments" / "multilingual"


def test_miracl_aggregation():
    assert test_registry.resolve_names("miracl") == [
        "miracl_en",
        "miracl_es",
        "miracl_fr",
        "miracl_de",
    ]
    assert MIRACL_KEYS["miracl_de"] == "io.github.project-miracl.de.dev"


def test_mldr_aggregation():
    resolved = test_registry.resolve_names("mldr")
    assert resolved == [f"mldr_{lang}" for lang in ("en", "es", "fr", "de", "it", "pt")]
    assert MLDR_KEYS["mldr_pt"] == "co.huggingface.datasets.mldr.pt.test"


def test_multilingual_ids_exist_in_datamaestro():
    from datamaestro.definitions import AbstractDataset

    for dataset_id in [*MIRACL_KEYS.values(), *MLDR_KEYS.values()]:
        assert AbstractDataset.find(dataset_id) is not None


def test_nanobeir11_multilingual_validation(monkeypatch):
    prepared = []

    def fake_prepare(dataset_id, download=None):
        prepared.append(dataset_id)
        dataset = MagicMock()
        dataset.documents.iter_documents.return_value = iter([None])
        dataset.topics.iter.return_value = iter([None])
        return dataset

    monkeypatch.setattr(validations, "prepare_collection", fake_prepare)
    monkeypatch.setattr(validations, "RandomFold", MagicMock())
    validations.nanobeir11_multilingual_validation_datasets.cache_clear()

    cfg = namedtuple("ValidationCfg", "seed size")(seed=0, size=500)
    folds, documents = validations.nanobeir11_multilingual_validation_datasets(cfg)
    validations.nanobeir11_multilingual_validation_datasets.cache_clear()

    # 11 datasets (no arguana, no touche) x 6 languages
    assert len(folds) == len(documents) == 11 * len(LIGHTON_LANGUAGES) == 66
    assert set(folds) == set(documents)
    assert not any("arguana" in name or "touche" in name for name in folds)
    assert {name.rsplit("_", 1)[1] for name in folds} == set(LIGHTON_LANGUAGES)
    assert "co.huggingface.nano-beir.nq.fr" in prepared
    assert all(
        dataset_id.rsplit(".", 1)[0] in NANO_BEIR_KEYS.values()
        for dataset_id in prepared
    )
    # Never MIRACL or MLDR: they are evaluation sets
    assert not any("miracl" in i or "mldr" in i for i in prepared)


def load_config(name: str) -> dict:
    """The YAML merged with its parents (parent first), as experimaestro does."""
    path = CONFIG_DIR / name
    data = yaml.full_load(path.read_text())
    parent = data.pop("parent", None)
    if parent is None:
        return data

    def merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    base_path = (path.parent / parent).resolve()
    base = load_config(str(base_path.relative_to(CONFIG_DIR, walk_up=True)))
    return merge(base, data)


CONFIGS = [
    "splade_first_stage.yaml",
    "mmbert_small_reranker.yaml",
    "mmbert_small_reranker_abl_exhausted_split.yaml",
    "mmicep_130m_l12+4.yaml",
    "mmicep_130m_l12+4_btnk_128.yaml",
]


@pytest.mark.parametrize("name", CONFIGS)
def test_configs_evaluate_on_miracl_only(name):
    cfg = load_config(name)
    evaluation = cfg["evaluation"]
    assert evaluation["datasets"] == ["miracl"]
    for suite in ("in_domain", "beir13", "beir13_decontaminated", "all_datasets"):
        assert not evaluation.get(suite, False)
    assert cfg["retriever"] == "naver/splade-v3-distilbert"
    assert cfg["retrieval"]["k"] == 100


@pytest.mark.parametrize("name", CONFIGS[1:])
def test_training_configs_use_lighton_data(name):
    cfg = load_config(name)
    assert cfg["learner"]["loss"] == "MSE_lighton_multilingual"
    assert cfg["learner"]["validation"] == Validation.NanoBEIR11Multilingual.value
    # Lengths are inherited from MICE-P, not overridden
    assert (cfg["max_query_length"], cfg["max_doc_length"]) == (1024, 1024)
    assert (cfg["plaid"]["max_query_length"], cfg["plaid"]["max_doc_length"]) == (
        32,
        512,
    )


def test_mmicep_phases_mirror_micep():
    phase1 = load_config("mmicep_130m_l12+4.yaml")
    phase2 = load_config("mmicep_130m_l12+4_btnk_128.yaml")
    for cfg, batch_size in ((phase1, 192), (phase2, 256)):
        assert cfg["learner"]["optimization"]["batch_size"] == batch_size
        assert cfg["learner"]["optimization"]["max_epochs"] == 5000
        assert cfg["learner"]["optimization"]["num_warmup_steps"] == 20000
        # PyYAML reads "5e-5" as a string (as in the MICE-P configs)
        lrs = cfg["grid_search"]["learner.optimization.lr"]
        assert [float(lr) for lr in lrs] == [1.5e-5, 5e-5]
        assert cfg["nb_repetitions"] == 2
        assert cfg["export_trained_models"] is True
        assert (cfg["query_prompt"], cfg["doc_prompt"]) == ("[Q] ", "[D] ")
        assert cfg["freeze_LN"] is True
        assert cfg["n_contextualization_layers"] == 12
        assert cfg["n_interaction_layers"] == 4
    assert phase2["from_experiment"] == phase1["id"]
    assert phase2["doc_bottleneck_dim"] == 128
    assert phase2["freeze_bottom_layers_phase1"] is True
    assert phase2["freeze_top_layers"] is False
    assert phase2["evaluation"]["eval_checkpoint"] == ["last", "best"]
    assert phase2["use_plaid"] is True
    assert phase2["plaid"]["n_bits"] == 4
