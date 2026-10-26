from unittest.mock import MagicMock, patch
import pytest

from configuration import Evaluation
from tests import (
    BEIR13_DECONTAMINATED_KEYS,
    BEIR_KEYS,
    INDOMAIN_KEYS,
    LIMIT_KEYS,
    LOTTE_KEYS,
    NANO_BEIR_KEYS,
    ROBUST04_KEYS,
    aggregations,
    aggregation_hf,
    build_tests,
    test_registry,
    TestRegistry,
)


class TestRegistryResolution:
    """Test resolution of dataset names, aliases, and aggregations."""

    def test_registered_dataset_count(self):
        # 3 in-domain + 13 beir + 13 beir_decontaminated + 5 lotte + 13 nano_beir + 2 limit + 1 robust04 = 50
        assert len(test_registry._datasets) == 50

    def test_resolve_individual_dataset(self):
        assert test_registry.resolve_names("fiqa") == ["fiqa"]
        assert test_registry.resolve_names(["fiqa", "scifact"]) == ["fiqa", "scifact"]

    def test_resolve_beir13(self):
        resolved = test_registry.resolve_names("beir13")
        assert resolved == list(BEIR_KEYS.keys())
        assert len(resolved) == 13

    def test_resolve_in_domain(self):
        resolved = test_registry.resolve_names("in_domain")
        assert resolved == list(INDOMAIN_KEYS.keys())
        assert len(resolved) == 3

    def test_resolve_beir13_decontaminated(self):
        resolved = test_registry.resolve_names("beir13_decontaminated")
        assert resolved == list(BEIR13_DECONTAMINATED_KEYS.keys())
        assert len(resolved) == 13

    def test_resolve_lotte(self):
        resolved = test_registry.resolve_names("lotte")
        assert resolved == list(LOTTE_KEYS.keys())
        assert len(resolved) == 5

    def test_resolve_nano_beir(self):
        resolved = test_registry.resolve_names("nano_beir")
        assert resolved == list(NANO_BEIR_KEYS.keys())
        assert len(resolved) == 13

    def test_resolve_limit_no_recursion(self):
        """'limit' is both an aggregation name and a dataset name.
        Verify that resolving 'limit' does not result in RecursionError and resolves both limit datasets.
        """
        resolved = test_registry.resolve_names("limit")
        assert resolved == ["limit", "limit_small"]

    def test_resolve_aliases_and_case_insensitivity(self):
        assert test_registry.resolve_names("limit-small") == ["limit_small"]
        assert test_registry.resolve_names("LIMIT") == ["limit", "limit_small"]
        assert test_registry.resolve_names("Nano BEIR") == list(NANO_BEIR_KEYS.keys())
        assert test_registry.resolve_names("BEIR13") == list(BEIR_KEYS.keys())

    def test_resolve_nested_paper_aggregation(self):
        resolved = test_registry.resolve_names("paper")
        expected = (
            list(INDOMAIN_KEYS.keys())
            + list(BEIR_KEYS.keys())
            + ["robust04"]
            + list(LOTTE_KEYS.keys())
        )
        assert resolved == expected

    def test_resolve_deduplication(self):
        # Requesting a dataset already part of an aggregation should not duplicate it
        resolved = test_registry.resolve_names(["beir13", "fiqa", "scifact"])
        assert resolved == list(BEIR_KEYS.keys())
        assert len(resolved) == len(set(resolved))

    def test_resolve_cycle_safety(self):
        reg = TestRegistry()
        reg.register_dataset("ds1", "path.to.ds1")
        reg.register_aggregation("agg_a", ["agg_b"])
        reg.register_aggregation("agg_b", ["agg_a", "ds1"])
        # Should terminate safely and resolve ds1
        assert reg.resolve_names("agg_a") == ["ds1"]

    def test_resolve_lotte_aliases(self):
        assert test_registry.resolve_names("lotte") == list(LOTTE_KEYS.keys())
        assert test_registry.resolve_names("lotte_search") == list(LOTTE_KEYS.keys())
        assert test_registry.resolve_names("lotte-search") == list(LOTTE_KEYS.keys())
        assert test_registry.resolve_names("Lotte-S") == list(LOTTE_KEYS.keys())

    def test_resolve_unknown_dataset_warns_loudly_with_suggestion(self, caplog):
        resolved = test_registry.resolve_names(["fiqa", "lotte_typo"])
        assert resolved == ["fiqa"]
        assert any("TEST REGISTRY WARNING" in record.message for record in caplog.records)
        assert any("Did you mean: 'lotte'" in record.message for record in caplog.records)


class TestDisplayAggregations:
    """Test exported aggregations dictionaries used by format.py and training_utils.py."""

    def test_aggregations_dictionary(self):
        assert "In Domain" in aggregations
        assert "BEIR13" in aggregations
        assert "BEIR13 Decontaminated" in aggregations
        assert "Lotte-S" in aggregations
        assert "Nano BEIR" in aggregations
        assert "LIMIT" in aggregations
        assert "Robust04" in aggregations
        assert "minified" in aggregations

        assert aggregations["In Domain"] == list(INDOMAIN_KEYS.keys())
        assert aggregations["BEIR13"] == list(BEIR_KEYS.keys())
        assert aggregations["BEIR13 Decontaminated"] == list(BEIR13_DECONTAMINATED_KEYS.keys())
        assert aggregations["Lotte-S"] == list(LOTTE_KEYS.keys())
        assert aggregations["Nano BEIR"] == list(NANO_BEIR_KEYS.keys())
        assert aggregations["LIMIT"] == list(LIMIT_KEYS.keys())

    def test_aggregation_hf_mapping(self):
        assert aggregation_hf["Mean In Domain"] == aggregations["In Domain"]
        assert aggregation_hf["BEIR 13"] == aggregations["BEIR13"]
        assert aggregation_hf["BEIR 13 (Decontaminated)"] == aggregations["BEIR13 Decontaminated"]
        assert aggregation_hf["LoTTE (OOD)"] == aggregations["Lotte-S"]
        assert aggregation_hf["Nano BEIR"] == aggregations["Nano BEIR"]
        assert aggregation_hf["LIMIT"] == aggregations["LIMIT"]


class TestLazyBuilding:
    """Test that building a collection only instantiates the requested datasets."""

    def test_lazy_building_only_requested_datasets(self):
        mock_eval = MagicMock()
        built_datasets = []

        def fake_build(self, *args, **kwargs):
            built_datasets.append(self.name)
            return mock_eval

        with patch("tests.DatasetSpec.build", side_effect=fake_build, autospec=True):
            collection = test_registry.build_collection(
                ["fiqa", "limit_small"],
                check_docs=False,
            )

            # Only fiqa and limit_small should have been built
            assert built_datasets == ["fiqa", "limit_small"]
            assert set(collection.collection.keys()) == {"fiqa", "limit_small"}

    def test_build_tests_with_empty_config_uses_minified(self):
        mock_eval = MagicMock()
        built_datasets = []

        def fake_build(self, *args, **kwargs):
            built_datasets.append(self.name)
            return mock_eval

        cfg = Evaluation()  # default has no datasets / flags active

        with patch("tests.DatasetSpec.build", side_effect=fake_build, autospec=True):
            collection = build_tests(cfg, check_docs=False)
            expected_minified = [
                "msmarco_dev",
                "trec2019",
                "trec2020",
                "scifact",
                "touche",
                "fiqa",
                "nfcorpus",
            ]
            assert built_datasets == expected_minified
            assert list(collection.collection.keys()) == expected_minified

    def test_build_tests_with_cherry_picked_datasets(self):
        mock_eval = MagicMock()
        built_datasets = []

        def fake_build(self, *args, **kwargs):
            built_datasets.append(self.name)
            return mock_eval

        cfg = Evaluation(datasets=["fiqa", "limit_small"])

        with patch("tests.DatasetSpec.build", side_effect=fake_build, autospec=True):
            collection = build_tests(cfg, check_docs=False)
            assert built_datasets == ["fiqa", "limit_small"]
            assert list(collection.collection.keys()) == ["fiqa", "limit_small"]

    def test_build_tests_with_limit_flag(self):
        mock_eval = MagicMock()
        built_datasets = []

        def fake_build(self, *args, **kwargs):
            built_datasets.append(self.name)
            return mock_eval

        cfg = Evaluation(limit=True)

        with patch("tests.DatasetSpec.build", side_effect=fake_build, autospec=True):
            collection = build_tests(cfg, check_docs=False)
            assert built_datasets == ["limit", "limit_small"]
            assert list(collection.collection.keys()) == ["limit", "limit_small"]

    def test_build_tests_combines_flags_and_cherry_pick(self):
        mock_eval = MagicMock()
        built_datasets = []

        def fake_build(self, *args, **kwargs):
            built_datasets.append(self.name)
            return mock_eval

        cfg = Evaluation(in_domain=True, datasets=["fiqa", "limit_small"])

        with patch("tests.DatasetSpec.build", side_effect=fake_build, autospec=True):
            collection = build_tests(cfg, check_docs=False)
            expected = ["msmarco_dev", "trec2019", "trec2020", "fiqa", "limit_small"]
            assert built_datasets == expected
            assert list(collection.collection.keys()) == expected
