"""mMICE-P training splits (src/multilingual_data.py)."""

import pytest

from multilingual_data import (
    LANGUAGES,
    SOURCES,
    split_identity,
    split_names,
    split_seed,
)


def test_default_splits_are_the_42_translated_ones():
    names = split_names()
    assert len(names) == len(LANGUAGES) * len(SOURCES) == 42
    assert len(set(names)) == 42
    assert names[:2] == ["fever_en", "fiqa_en"]
    assert {split_identity(n)[1] for n in names} == {"en", "fr", "es", "it", "de", "pt"}
    assert not any(n.startswith(("miracl_", "mldr_")) for n in names)


def test_language_selection():
    assert split_names(("de",), ("fiqa", "nq")) == ["fiqa_de", "nq_de"]


@pytest.mark.parametrize("source", ["miracl", "mldr"])
def test_evaluation_sources_are_refused(source):
    with pytest.raises(ValueError, match="evaluation sets"):
        split_names(("en",), ("fiqa", source))


def test_unknown_language_or_source_raises():
    with pytest.raises(ValueError, match="Unknown language"):
        split_names(("xx",))
    with pytest.raises(ValueError, match="No 'wiki' split"):
        split_names(("fr",), ("wiki",))
    with pytest.raises(ValueError, match="non-empty"):
        split_names(())


def test_split_identity():
    assert split_identity("squadv2_pt") == ("squadv2", "pt")


def test_split_seeds_differ_across_translations_and_are_stable():
    seeds = {split_seed(42, n) for n in split_names()}
    assert len(seeds) == 42
    assert (
        split_seed(42, "fiqa_fr")
        == split_seed(42, "fiqa_fr")
        != split_seed(43, "fiqa_fr")
    )
    assert 0 <= split_seed(42, "fiqa_fr") < 2**32
