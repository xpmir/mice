"""Configuration surface of the LightOn multilingual training data (mmicep_brief.md, step 2)."""

import pytest
from configuration import (
    LIGHTON_LANGUAGES,
    CE_FineTuning,
    ExhaustedSplit,
    Losses,
)
from experimaestro.experiments.grid import generate_grid
from format import loss_names
from multilingual_data import LANGUAGES


def test_defaults():
    cfg = CE_FineTuning()
    assert cfg.lighton_languages == ["en", "fr", "es", "it", "de", "pt"]
    assert cfg.lighton_alpha == 1.0
    assert cfg.lighton_exhausted_split == ExhaustedSplit.REPLAY.value == "replay"
    # The Ettin data selector is unchanged
    assert cfg.ettin_subset_exclude == []


def test_default_languages_are_not_shared_between_instances():
    a, b = CE_FineTuning(), CE_FineTuning()
    a.lighton_languages.append("xx")
    assert b.lighton_languages == list(LIGHTON_LANGUAGES)


def test_single_language_list():
    assert CE_FineTuning(lighton_languages=["de"]).lighton_languages == ["de"]


def test_new_loss_and_its_label():
    assert Losses("MSE_lighton_multilingual") is Losses.MSE_lighton_multilingual
    assert Losses("MSE_mixedbread_large") is Losses.MSE_mixedbread_large
    assert loss_names["MSE_lighton_multilingual"] == "MSE (LightOn multilingual)"


def test_exhausted_split_values():
    assert {e.value for e in ExhaustedSplit} == {"replay", "reshuffle"}
    with pytest.raises(ValueError):
        ExhaustedSplit("restart")


def test_data_module_uses_the_same_languages():
    assert LANGUAGES == LIGHTON_LANGUAGES


def test_grid_over_exhausted_split_and_alpha():
    cfg = CE_FineTuning(
        grid_search={
            "lighton_exhausted_split": ["replay", "reshuffle"],
            "lighton_alpha": [1.0, 0.5],
        }
    )
    configs, _ = generate_grid(cfg)
    assert {(c.lighton_exhausted_split, c.lighton_alpha) for c in configs} == {
        ("replay", 1.0),
        ("replay", 0.5),
        ("reshuffle", 1.0),
        ("reshuffle", 0.5),
    }
