"""Training data for mMICE-P (mmicep_brief.md, step 1).

LightOn's ``lightonai/embeddings-fine-tuning-filtered-<lang>`` translated splits,
as pointwise (query, document, ``rerank_scores`` teacher score) samples. As for
MICE-P (:func:`training_utils.build_ettin_reranker_v1_data`), the samples are
shuffled at the (query, document) level with a fixed seed and saved to disk once;
unlike MICE-P, every split is kept apart (one directory per split) so that a
batch can be drawn from a single split (LightOn's recipe, step 3).
"""

import gc
import hashlib
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from datamaestro_ir.config.ai.lighton.embeddings_fine_tuning import SOURCES_BY_LANG
from datamaestro_ir.data.lighton import EmbeddingsFineTuningPointwiseSamples
from experimaestro import Param, Task, pathgenerator
from experimaestro.launcherfinder import find_launcher

from configuration import LIGHTON_LANGUAGES
from samplers import prepare_collection

logger = logging.getLogger(__name__)

LANGUAGES = LIGHTON_LANGUAGES
"""Training languages (brief: decisions already taken)."""

SOURCES = ("fever", "fiqa", "hotpotqa", "msmarco", "nq", "squadv2", "trivia")
"""The translated sources, available in every language."""

EVALUATION_SOURCES = frozenset({"miracl", "mldr"})
"""Sources that are evaluation sets: training on them would leak."""


def split_names(languages=LANGUAGES, sources=SOURCES) -> list[str]:
    """``<source>_<lang>`` split names, language-major. Raises on an evaluation
    source, an unknown language, or a source missing for a language."""
    if not languages or not sources:
        raise ValueError("No training split: languages and sources must be non-empty")
    leaking = EVALUATION_SOURCES.intersection(sources)
    if leaking:
        raise ValueError(
            f"{sorted(leaking)} are evaluation sets and cannot be trained on"
        )
    names = []
    for lang in languages:
        if lang not in SOURCES_BY_LANG:
            raise ValueError(
                f"Unknown language {lang!r}: expected one of {sorted(SOURCES_BY_LANG)}"
            )
        for source in sources:
            if source not in SOURCES_BY_LANG[lang]:
                raise ValueError(f"No {source!r} split for language {lang!r}")
            names.append(f"{source}_{lang}")
    return names


def split_seed(seed: int, name: str) -> int:
    """Shuffling seed of split ``name``: derived from ``seed`` and the name, so that
    the translations of a source (aligned row for row across languages) are not
    shuffled into the same order. Stable across runs and Python versions."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def split_identity(name: str) -> tuple[str, str]:
    """``"fiqa_fr"`` -> ``("fiqa", "fr")``."""
    source, lang = name.rsplit("_", 1)
    return source, lang


class ShufflePointwiseSplits(Task):
    """Shuffles each pointwise split and saves it to its own directory.

    One task for all splits (one entry in the job list); its outputs are copies of
    the input configurations with ``local_path`` set to their directory, in the
    same order as :attr:`samples`. Splits already saved are skipped when the task
    is restarted.
    """

    samples: Param[list[EmbeddingsFineTuningPointwiseSamples]]
    """The pointwise splits, each with a distinct ``name``."""

    seed: Param[int]
    """Base shuffling seed; split ``name`` uses :func:`split_seed` (seed, name)."""

    output_dir: Annotated[Path, pathgenerator("splits")]
    """One sub-directory per split, named after it."""

    def __validate__(self):
        names = [s.name for s in self.samples]
        if not names:
            raise ValueError("ShufflePointwiseSplits received no split")
        if len(set(names)) != len(names):
            raise ValueError(f"Split names must be distinct: {names}")

    def task_outputs(self, dep) -> list[EmbeddingsFineTuningPointwiseSamples]:
        outputs = []
        for sample in self.samples:
            params = {
                k: getattr(sample, k) for k in sample.__xpmtype__.arguments.keys()
            }
            params["local_path"] = self.output_dir / sample.name
            params["streaming"] = False
            outputs.append(dep(sample.__class__.C(**params)))
        return outputs

    def execute(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(self.samples):
            target = self.output_dir / sample.name
            if (target / "dataset_info.json").exists():
                logger.info(
                    "[%d/%d] %s: already saved", i + 1, len(self.samples), sample.name
                )
                continue
            data = sample.data
            seed = split_seed(self.seed, sample.name)
            logger.info(
                "[%d/%d] %s: shuffling %d rows (seed=%d)",
                i + 1,
                len(self.samples),
                sample.name,
                len(data),
                seed,
            )
            # Write to a temporary directory, so that a crash never leaves a
            # partial split that would then be skipped on restart.
            tmp = target.with_name(target.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            # Single process: the joined split is an in-memory table, which
            # save_to_disk(num_proc > 1) would copy into every worker (slower).
            data.shuffle(seed=seed).save_to_disk(tmp)
            tmp.rename(target)
            # The joined split is held in memory: release it before the next one
            del data
            sample.__dict__.pop("data", None)
            gc.collect()


@lru_cache(maxsize=None)
def build_lighton_pointwise_splits(
    languages: tuple[str, ...] = LANGUAGES,
    seed: int = 42,
    requirements: str = None,
    download: bool = True,
) -> dict[str, EmbeddingsFineTuningPointwiseSamples]:
    """The shuffled pointwise splits ``{name: dataset}`` for ``languages`` (all
    :data:`SOURCES`), from a single :class:`ShufflePointwiseSplits` task.
    ``seed`` defaults to 42, as ``build_trainer`` does for the Ettin data."""
    names = split_names(languages)
    samples = [
        prepare_collection(
            f"ai.lighton.embeddings_fine_tuning[name={name},format=pointwise]",
            download=download,
        )
        for name in names
    ]
    outputs = ShufflePointwiseSplits.C(samples=samples, seed=seed).submit(
        launcher=find_launcher(requirements)
    )
    return dict(zip(names, outputs))
