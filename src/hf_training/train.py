import logging
from typing import Any, Dict, NamedTuple
from datasets import load_dataset, load_from_disk

import numpy as np
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses.MarginMSELoss import MarginMSELoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

from experimaestro.launcherfinder import find_launcher
from pathlib import Path
from experimaestro import (
    Config,
    Task,
    Param,
    pathgenerator,
    Annotated,
    Meta,
    tag,
)
import torch
from xpmir.utils.utils import EasyLogger
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from hf_training.configuration import HFFinetuning
from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper
from xpmir.learning.optim import Module, ModuleLoader
from xpmir.neural.cross import MiniLMCrossScorer
from xpmir.text.huggingface import (
    HFCLSEncoder, 
    HFStringTokenizer,
    HFStringTokenizer,
)
from xpmir.text import TokenizedTextEncoder
from hf_training.utils import XPMIRCrossEncoder
from tests import minified_tests, paper_tests
from xpmir.neural.huggingface import HFCrossScorer

from transformers import AutoConfig

from functools import partial

from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.rankers.standard import BM25

import xpmir.interfaces.anserini as anserini
from datamaestro_text.data.ir import Documents
from xpmir.rankers import scorer_retriever, document_cache, Retriever


class DataOutput(Config):
    train_path: Param[Path]

    eval_path: Param[Path]

    sample_max: Param[int] = 0

class Dataloader(Task):
    sample_max: Param[int] = 2_000_000
    """Maximum number of samples considered (before shuffling). 0 for no limit."""

    test_size: Param[int] = 10_000

    save_path: Annotated[Path, pathgenerator("data")]
    """Path where to store the dataset"""

    def task_outputs(self, dep) -> DataOutput:
        return dep(
            DataOutput.C(
                train_path = self.save_path / "ms-marco-margin-mse-train",
                eval_path = self.save_path / "ms-marco-margin-mse-eval",
                sample_max = self.sample_max,
            )
        )

    def execute(self):
        logging.info("The dataset has not been fully stored as texts on disk yet. We will do this now.")
        corpus = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
        corpus = dict(zip(corpus["passage_id"], corpus["passage"]))
        queries = load_dataset("sentence-transformers/msmarco", "queries", split="train")
        queries = dict(zip(queries["query_id"], queries["query"]))
        dataset = load_dataset("sentence-transformers/msmarco", "bert-ensemble-margin-mse", split="train")
        dataset = dataset.select(range(self.sample_max))

        def id_to_text_map(batch):
            return {
                "query": [queries[qid] for qid in batch["query_id"]],
                "positive": [corpus[pid] for pid in batch["positive_id"]],
                "negative": [corpus[pid] for pid in batch["negative_id"]],
                "score": batch["score"],
            }

        dataset = dataset.map(id_to_text_map, batched=True, remove_columns=["query_id", "positive_id", "negative_id"])
        dataset = dataset.train_test_split(test_size=self.test_size)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        train_dataset.save_to_disk(self.save_path / "ms-marco-margin-mse-train")
        eval_dataset.save_to_disk(self.save_path / "ms-marco-margin-mse-eval")
        logging.info(
            "The dataset has now been stored as texts on disk."
        )
    
class LearnerOutput(NamedTuple):
    learned_model: ModuleLoader

    checkpoints: Dict[str, Any]

class HFLearner(Task, EasyLogger):
    """Model Learner

    The learner task is generic, and takes two main arguments: (1) the model
    defines the model (e.g. DRMM), and (2) the trainer defines how the model
    should be trained (e.g. pointwise, pairwise, etc.)

    When submitted, it returns a dictionary based on the `listeners`
    """
    # Training
    model: Param[str]
    """Name of the pretrained model to be fine-tuned"""

    xpmirModule: Param[Module]
    """The XPMIR module to be used for the model"""

    batch_size: Param[int] = 64
    """Batch size used during training"""

    num_epochs: Param[int] = 1
    """Number of epochs"""

    lr: Param[float] = 8.0e-6
    """Learning rate"""

    warmup_ratio: Param[float] = 0.1
    """Warmup ratio for learning rate scheduler"""

    max_grad_norm: Param[float] = 1.0
    """Maximum gradient norm for gradient clipping"""

    weight_decay: Param[float] = 0.0
    """Weight decay for optimizer"""

    adam_epsilon: Param[float] = 1.0e-8
    """Epsilon for Adam optimizer"""

    use_fp16: Param[bool] = False
    """Use mixed precision when training"""
    
    use_bf16: Param[bool] = False
    """Use bfloat16 when training"""

    checkpoint_steps: Param[int] = 1
    """Number of steps between each checkpoint"""

    validation_steps: Param[int] = 1
    """Number of steps between each validation"""

    validation_top_k: Param[int] = 1000
    """Number of top-k documents to consider during validation"""

    data_output: Param[DataOutput]

    seed: Param[int] = 0
    """The seed used for experiments"""

    full_deterministic_gpu: Param[bool] = False
    """Make GPU operations fully deterministic (may be slower)"""

    logging_steps: Meta[int] = 1
    """Number of steps between each log"""

    dataloader_num_workers: Meta[int] = 4
    """Number of workers for data loading"""

    output_dir: Annotated[Path, pathgenerator("outputs")]
    """The path to the checkpoints"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device(s) to be used for the model"""

    def task_outputs(self, dep) -> LearnerOutput:
        return LearnerOutput(
                learned_model=dep(
                    ModuleLoader.C(
                        value=self.xpmirModule,
                        path=self.output_dir / "final",
                    )
                ),
                checkpoints={interval: dep(ModuleLoader.C(value=self.xpmirModule, path=self.output_dir / f"checkpoint-{interval}")) for interval in range(0, self.data_output.sample_max, self.checkpoint_steps)} ,
            )   

    def execute(self):
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        model = CrossEncoder(self.model)
        print("Model max length:", model.max_length)
        print("Model num labels:", model.num_labels)

        # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/sentence-transformers/msmarco
        logging.info("Read train dataset")
        try:
            train_dataset = load_from_disk(self.data_output.train_path)
            eval_dataset = load_from_disk(self.data_output.eval_path)
        except FileNotFoundError:
            raise FileNotFoundError("The dataset has not been fully stored as texts on disk yet.")
        
        logging.info(train_dataset)
        # 3. Define our training loss
        loss = MarginMSELoss(model)

        # 4. Define the evaluator. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
        evaluator = CrossEncoderNanoBEIREvaluator(
            dataset_names=["msmarco", "nfcorpus", "nq"], 
            batch_size=self.batch_size,
            rerank_k=self.validation_top_k,
        )
        evaluator(model)

        # 5. Define the training arguments
        args = CrossEncoderTrainingArguments(
            # Required parameter:
            output_dir=self.output_dir,
            # Optional training parameters:
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.lr,  # Lower than usual
            warmup_ratio=self.warmup_ratio,
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            adam_epsilon=self.adam_epsilon,
            fp16=self.use_fp16,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=self.use_bf16,  # Set to True if you have a GPU that supports BF16
            load_best_model_at_end=True,
            metric_for_best_model=f"eval_NanoBEIR_R{self.validation_top_k}_mean_ndcg@10",
            # Optional tracking/debugging parameters:
            eval_strategy="steps",
            eval_steps=self.validation_steps,
            save_strategy="steps",
            save_steps=self.checkpoint_steps,
            save_total_limit=10,
            logging_steps=self.logging_steps,
            logging_first_step=True,
            seed=self.seed,
            dataloader_num_workers=self.dataloader_num_workers,
            full_determinism=self.full_deterministic_gpu,
        )

        # 6. Create the trainer & start training
        trainer = CrossEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
        )
        trainer.train()

        # 7. Evaluate the final model, useful to include these in the model card
        evaluator(model)

        # 8. Save the final model
        final_output_dir = self.output_dir / "final"
        model.save_pretrained(final_output_dir)


@ir_experiment()
def run(helper: IRExperimentHelper, cfg: HFFinetuning) -> PaperResults:
    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    launcher_learner = find_launcher(cfg.learner.requirements) 
    launcher_loader = find_launcher(cfg.loader_requirements)

    device = cfg.device

    launcher_evaluate = find_launcher(cfg.evaluation.retrieval.requirements)
    launcher_index = find_launcher(cfg.evaluation.indexation.requirements)

    if cfg.evaluation.all_datasets:
        tests = paper_tests(cfg.evaluation.test_max_topics)
    else:
        tests = minified_tests(cfg.evaluation.test_max_topics)
        
    basemodel = BM25.C()

    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.evaluation.retrieval.batch_size,
        batcher=PowerAdaptativeBatcher.C(),
        device=cfg.device,
    ) #: Model-based retrievers

    @document_cache
    def index_builder(documents: Documents):
        return anserini.IndexCollection.C(
            documents=documents,
        ).submit(launcher=launcher_index,)

    def retriever(
        name,
        model,
        documents: Documents,
    ) -> Retriever.C:
        return anserini.AnseriniRetriever.C(
            index=index_builder()(documents),
            model=model,
            k=cfg.evaluation.retrieval.k,
            store=documents,
        ).tag("first-stage", name)

    bm25_retriever = partial(retriever, "bm25", basemodel)
    tests.evaluate_retriever(bm25_retriever)

    xpmir_scorer: MiniLMCrossScorer = MiniLMCrossScorer.C(
        encoder=TokenizedTextEncoder.C(
            tokenizer=HFStringTokenizer.from_pretrained_id(cfg.model_id),
            encoder=HFCLSEncoder.from_pretrained_id(cfg.model_id),
        )
    ).tag("scorer", cfg.id)

    paper_results = list()
    config = AutoConfig.from_pretrained(cfg.model_id)
    xpmirModule = XPMIRCrossEncoder.C(
        hf_id=cfg.model_id,
        max_length=config.max_position_embeddings,
    )

    for i in range(cfg.nb_repetitions):
        seed = np.random.RandomState(cfg.seed + i).randint((2**32) - 1)
        # 1. Prepare the dataset
        data_output = Dataloader.C(
            sample_max=cfg.learner.sample_max,
            test_size=cfg.test_size,
        ).submit(launcher=launcher_loader)

        learner = HFLearner.C(
            model=cfg.model_id,
            xpmirModule=xpmirModule,
            batch_size=cfg.learner.optimization.batch_size,
            num_epochs=cfg.learner.optimization.num_epochs,
            lr=cfg.learner.optimization.lr,
            warmup_ratio=cfg.learner.optimization.warmup_ratio,
            max_grad_norm=cfg.learner.max_grad_norm,
            weight_decay=cfg.learner.optimization.weight_decay,
            adam_epsilon=cfg.learner.optimization.eps,
            use_fp16=cfg.learner.use_fp16,
            use_bf16=cfg.learner.use_bf16,
            checkpoint_steps=cfg.learner.checkpoint_steps,
            validation_steps=cfg.learner.validation_steps,
            validation_top_k=cfg.learner.validation_top_k,
            logging_steps=cfg.learner.logging_steps,
            dataloader_num_workers=cfg.learner.dataloader_num_workers,
            data_output=data_output,
            seed=tag(seed),
            full_deterministic_gpu=cfg.deterministic_gpu,
        )

        learner_output = learner.submit(launcher=launcher_learner)
        
        load_model = learner_output.learned_model

        tests.evaluate_retriever(
            partial(
                model_based_retrievers,
                scorer=xpmir_scorer.tag("model", cfg.model_id),
                retrievers=bm25_retriever,
                device=device,
            ),
            launcher=launcher_evaluate,
            model_id=f"{cfg.model_id}-zs-RR@10-{seed}",
            init_tasks=[load_model],
        )

    # Wait for all the experiments in the loop to finish before processing the dataframes
    helper.xp.wait()

    df = tests.to_dataframe()
    metric_cols = [("metric", "AP"), ("metric", "RR@10"), ("metric", "nDCG@10")]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, downcast="float")
    df_grouped = (
        df
        .groupby(["dataset", ("tag", "first-stage"), ("tag", "model"), ("tag", "scorer")], dropna=False)[metric_cols]
        .agg(['mean', 'var'])
        .reset_index()
    )
    logging.info(df_grouped)

