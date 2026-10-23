"""
Unit tests for experiment model resolver, decoupled config validation, and parameter override warnings.
Run with: `uv run pytest tests/test_experiment_model_resolver.py -v`
"""

import logging
from pathlib import Path
import pytest

from training_utils import (
    find_candidate_experiment_dirs,
    find_experiment_results_model_dir,
    parse_experiment_ref,
    resolve_trained_model_loader,
    validate_and_align_config,
)
from configuration import CE_FineTuning
from MICE.experiments.micev2_training import MiceV2_FineTuning, build_mice_v2


def test_find_experiment_results_model_dir_direct_path(tmp_path):
    """Test resolving a direct path to a model folder containing experimaestro.json."""
    model_dir = tmp_path / "results" / "models" / "TestModel"
    model_dir.mkdir(parents=True)
    (model_dir / "experimaestro.json").write_text("{}")

    # Pass the direct model directory
    resolved = find_experiment_results_model_dir(str(model_dir))
    assert resolved == model_dir

    # Pass the parent run directory
    resolved_run = find_experiment_results_model_dir(str(tmp_path))
    assert resolved_run == model_dir


def test_find_experiment_results_model_dir_not_found():
    """Test that a non-existent experiment raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found in any workspace"):
        find_experiment_results_model_dir("non_existent_experiment_xyz_12345")


def test_resolve_trained_model_loader_fallback_to_hub():
    """Test that a Hugging Face repo ID candidate that is not a local experiment falls back to Hub."""
    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    assert loader is not None
    assert hasattr(loader, "model")
    assert loader.model.hf_id == "cross-encoder/ettin-reranker-68m-v1"


def test_resolve_local_experiment_if_present():
    """Test resolving local experiment 'train_mice_ettin32m_l4+4' if present on the machine."""
    try:
        resolved = find_experiment_results_model_dir("train_mice_ettin32m_l4+4")
        assert resolved.exists()
        assert (resolved / "experimaestro.json").exists()
        assert "Mice-l4+4-ettin-32m" in str(resolved)

        loader = resolve_trained_model_loader("train_mice_ettin32m_l4+4")
        assert loader is not None
        assert hasattr(loader, "model")
        assert loader.model.hf_id == "jhu-clsp/ettin-encoder-32m"
        assert loader.model.n_contextualization_layers == 4
        assert loader.model.n_interaction_layers == 4
    except FileNotFoundError:
        pytest.skip("train_mice_ettin32m_l4+4 experiment not available on this machine")


def test_ce_finetuning_from_experiment_field():
    """Test CE_FineTuning configuration supports from_experiment."""
    cfg = CE_FineTuning(from_experiment="train_mice_ettin32m_l4+4")
    assert cfg.from_experiment == "train_mice_ettin32m_l4+4"


def test_init_micev2_from_experiment_checkpoint():
    """Test InitMICEV2FromCheckpoint execution using an exported model loader from an experiment/hub."""
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        utility_pred=True,
        hard_pruning=True,
        hard_prune_threshold=0.5,
    )
    for task in init_tasks:
        task.instance().execute()

    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
    ).instance()

    init_task.execute()

    scorer = init_task.scorer
    assert scorer._initialized
    for param in scorer.hf_model.utility_predictor.parameters():
        assert not param.requires_grad
    for param in scorer.hf_model.model.layers.parameters():
        assert not param.requires_grad
    for param in scorer.hf_model.top_layers.parameters():
        assert param.requires_grad


def test_validate_and_align_inherits_default_parameters(caplog):
    """Test that default parameters inherit from the loaded model with INFO logs."""
    cfg = MiceV2_FineTuning(from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m")
    tags = {}
    loader = resolve_trained_model_loader(cfg.from_experiment)

    with caplog.at_level(logging.INFO):
        validate_and_align_config(cfg, loader.model, tags=tags, from_ref=cfg.from_experiment)

    # Verified inheritance: 10 contextualization and 4 interaction layers inherited from 68m model
    assert cfg.base == "cross-encoder/ettin-reranker-68m-v1"
    assert cfg.n_contextualization_layers == 10
    assert cfg.n_interaction_layers == 4
    assert tags["base"] == "cross-encoder/ettin-reranker-68m-v1"
    assert tags["from_experiment"] == "VictorMorand/MiceV2-l10p4-ettin-68m"

    assert "[Model Alignment] Inherited base" in caplog.text
    assert "[Model Alignment] Inherited 'n_contextualization_layers'" in caplog.text
    assert "[PARAM OVERRIDE]" not in caplog.text


def test_validate_and_align_warns_on_explicit_override(caplog):
    """Test that explicitly set parameters differing from loaded model emit loud WARNINGs."""
    cfg = MiceV2_FineTuning(
        from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m",
        base="jhu-clsp/ettin-encoder-17m",
        n_interaction_layers=2,  # Loaded model has 4
    )
    tags = {}
    loader = resolve_trained_model_loader(cfg.from_experiment)

    with caplog.at_level(logging.INFO):
        validate_and_align_config(cfg, loader.model, tags=tags, from_ref=cfg.from_experiment)

    # User-specified values should be preserved
    assert cfg.base == "jhu-clsp/ettin-encoder-17m"
    assert cfg.n_interaction_layers == 2

    # Loud warnings must be logged
    assert "[PARAM OVERRIDE] Config explicitly overrides base model!" in caplog.text
    assert (
        "[PARAM OVERRIDE] Config explicitly overrides loaded model parameter 'n_interaction_layers'!"
        in caplog.text
    )


def test_build_mice_v2_with_from_experiment():
    """Test that build_mice_v2 automatically loads checkpoint and aligns config when from_experiment is set."""
    cfg = MiceV2_FineTuning(from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m")
    tags = {}

    scorer, init_tasks = build_mice_v2(cfg, cfg_tags=tags)

    # Checks base and tags alignment
    assert cfg.base == "cross-encoder/ettin-reranker-68m-v1"
    assert tags["base"] == "cross-encoder/ettin-reranker-68m-v1"
    assert tags["from_experiment"] == "VictorMorand/MiceV2-l10p4-ettin-68m"

    # Checks init task is InitMICEV2FromCheckpoint
    assert len(init_tasks) == 1
    assert "InitMICEV2FromCheckpoint" in init_tasks[0].__class__.__name__


def test_modernbert_mice_scorer_warns_on_override_from_checkpoint(caplog):
    """Test that ModernBertMiceScorer.__initialize__ logs [MICE Config Override] warnings."""
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="VictorMorand/MiceV2-l10p4-ettin-68m",
        n_contextualization_layers=10,
        n_interaction_layers=4,
        top_layers_doc_ln=True,
        bottom_final_norm=False,
        utility_temp=2.0,  # Overriding checkpoint's 1.0
        hard_prune_threshold=0.3,  # Overriding checkpoint's 0.5
    )

    with caplog.at_level(logging.WARNING):
        for task in init_tasks:
            task.instance().execute()

    assert "[MICE Config Override] Overriding checkpoint parameter" in caplog.text
    assert "utility_temp" in caplog.text
    assert "hard_prune_threshold" in caplog.text


def test_modernbert_mice_scorer_structural_mismatch_raises(caplog):
    """Test that overriding structural compress_dim logs warnings and fails weight loading."""
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    scorer_cfg, init_tasks = mice_v2_scorer(
        hf_id="VictorMorand/MiceV2-l10p4-ettin-68m",
        n_contextualization_layers=10,
        n_interaction_layers=2,  # Overriding checkpoint's 4 layers
        compress_dim=2.0,  # Overriding checkpoint's 1.0 compression
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            for task in init_tasks:
                task.instance().execute()

    assert "[MICE Config Override] Overriding checkpoint parameter" in caplog.text
    assert "compress_dim" in caplog.text


def test_checkpoint_loader_taskstub_job_serialization():
    """Test that an init task with a loaded checkpoint_loader (containing a TaskStub)
    can be serialized for job execution via state_dict/outputjson without error."""
    from experimaestro import state_dict
    from experimaestro.core.context import SerializationContext
    from experimaestro.core.objects.config import TaskStub
    from MICE.experiments.micev2_training import MiceV2_FineTuning, build_mice_v2

    cfg = MiceV2_FineTuning(from_experiment="VictorMorand/MiceV2-l10p4-ettin-68m")
    scorer, init_tasks = build_mice_v2(cfg, cfg_tags={})

    assert len(init_tasks) == 1
    init_task = init_tasks[0]
    checkpoint_loader = init_task.checkpoint_loader

    # Verify checkpoint_loader.task is a TaskStub
    assert isinstance(checkpoint_loader.__xpm__.task, TaskStub)

    # Test serialization of init_task (as happens when scheduler writes params.json)
    context = SerializationContext(save_directory=None)
    init_task.__xpm__.seal(context)
    data = state_dict(context, [init_task])
    assert data is not None

    # Verify TaskStub was serialized with stub=True
    stub_objects = [obj for obj in data["objects"] if obj.get("stub")]
    assert len(stub_objects) == 1
    assert stub_objects[0]["id"] == id(checkpoint_loader.__xpm__.task)


def test_init_micev2_from_checkpoint_with_compression():
    """Test that InitMICEV2FromCheckpoint discards top layer seeding when compress_dim > 1.0,
    loading bottom layers from the uncompressed checkpoint and keeping compressed top layers
    at fresh random initialization without PyTorch shape mismatch errors."""
    import torch
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    # 1. Resolve uncompressed checkpoint (VictorMorand/MiceV2-l10p4-ettin-68m has 512-dim layers)
    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    # 2. Build target model with compress_dim = 1.333333333 (top layers hidden_size = 384)
    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.333333333,
        utility_pred=True,
    )

    # 3. Create InitMICEV2FromCheckpoint task
    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
    ).instance()

    # 4. Execute: must succeed without size mismatch exception
    init_task.execute()

    scorer = init_task.scorer
    assert scorer._initialized
    hf_model = scorer.hf_model

    # Bottom layers: 512-dim, frozen
    assert hf_model.model.layers[0].attn.Wqkv.weight.shape == torch.Size([1536, 512])
    for param in hf_model.model.layers.parameters():
        assert not param.requires_grad
    for param in hf_model.model.embeddings.parameters():
        assert not param.requires_grad

    # Top interaction layers: 384-dim, trainable
    assert hf_model.top_layers[0].self_attn.Wqkv.weight.shape == torch.Size([1152, 384])
    assert hf_model.adapter.weight.shape == torch.Size([384, 512])
    assert hf_model.output_modules.dense_score.linear.weight.shape == torch.Size([1, 384])
    for param in hf_model.top_layers.parameters():
        assert param.requires_grad
    assert hf_model.adapter.weight.requires_grad
    assert hf_model.output_modules.dense_score.linear.weight.requires_grad


def test_init_micev2_from_checkpoint_with_compress_init_random_uncompressed():
    """Test that compress_dim=1.0 with compress_init_option='random' re-initializes top layers
    and head with fresh random weights while preserving checkpoint bottom layers."""
    import torch
    from safetensors.torch import load_file
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.0,
        compress_init_option="random",
        utility_pred=True,
    )

    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
    ).instance()

    init_task.execute()

    scorer = init_task.scorer
    assert scorer._initialized
    hf_model = scorer.hf_model

    sd = load_file(loader.path)

    # 1. Bottom layers: MUST match checkpoint and be frozen
    assert torch.equal(hf_model.model.layers[0].attn.Wqkv.weight, sd["model.layers.0.attn.Wqkv.weight"])
    for param in hf_model.model.layers.parameters():
        assert not param.requires_grad

    # 2. Top interaction layers and classification head: MUST NOT match checkpoint (fresh random weights) and be trainable
    assert not torch.equal(hf_model.top_layers[0].self_attn.Wqkv.weight, sd["top_layers.0.self_attn.Wqkv.weight"])
    for param in hf_model.top_layers.parameters():
        assert param.requires_grad
    assert hf_model.output_modules.dense_score.linear.weight.requires_grad

    # 3. Tokenizer: MUST be initialized in main process and functional
    assert scorer.tokenizer._initialized
    assert hasattr(scorer.tokenizer, "tokenizer") and scorer.tokenizer.tokenizer is not None

    from xpmir.letor.records import PairwiseItems, PairwiseItem
    from types import SimpleNamespace
    p_item = PairwiseItem(
        query={"id": "q1", "text_item": SimpleNamespace(text="test query")},
        positive={"id": "d1", "text_item": SimpleNamespace(text="test doc 1")},
        negative={"id": "d2", "text_item": SimpleNamespace(text="test doc 2")},
    )
    tokenized = scorer.batch_tokenize(PairwiseItems([p_item]))
    assert tokenized.tokenized_q.ids.shape[0] == 2
    assert tokenized.tokenized_docs.ids.shape[0] == 2


def test_init_micev2_from_checkpoint_with_compress_init_seed_uncompressed():
    """Test that compress_dim=1.0 with compress_init_option='seed' preserves all checkpoint weights (top and bottom)."""
    import torch
    from safetensors.torch import load_file
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.0,
        compress_init_option="seed",
        utility_pred=True,
    )

    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
    ).instance()

    init_task.execute()

    scorer = init_task.scorer
    hf_model = scorer.hf_model

    sd = load_file(loader.path)

    # Bottom layers, top layers, and classifier must all match checkpoint
    assert torch.equal(hf_model.model.layers[0].attn.Wqkv.weight, sd["model.layers.0.attn.Wqkv.weight"])
    assert torch.equal(hf_model.top_layers[0].self_attn.Wqkv.weight, sd["top_layers.0.self_attn.Wqkv.weight"])
    assert torch.equal(hf_model.output_modules.dense_score.linear.weight, sd["output_modules.dense_score.linear.weight"])


def test_load_model_raises_on_shape_mismatch():
    """Test that ModernBertMiceScorer.load_model strictly raises RuntimeError on weight shape mismatch."""
    import tempfile
    import torch
    from safetensors.torch import save_file
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    scorer_cfg, _ = mice_v2_scorer(
        hf_id="VictorMorand/MiceV2-l10p4-ettin-68m",
        n_contextualization_layers=10,
        n_interaction_layers=4,
        compress_dim=1.0,
    )
    scorer = scorer_cfg.instance()
    scorer.initialize()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create incompatible weights: layer 0 attention weight has wrong shape [100, 100] instead of [1536, 512]
        bad_sd = {"model.layers.0.attn.Wqkv.weight": torch.randn(100, 100)}
        save_file(bad_sd, str(tmp_path / "model.safetensors"))

        with pytest.raises(RuntimeError) as excinfo:
            scorer.load_model(tmp_path)
        assert "size mismatch" in str(excinfo.value)


def test_load_model_ignore_top():
    """Test that ModernBertMiceScorer.load_model with ignore_top=True discards top weights and succeeds."""
    import tempfile
    import torch
    from safetensors.torch import save_file
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model
    uncompressed_model_dir = Path(loader.path).parent

    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.333333333,
    )
    scorer = scorer_cfg.instance()
    scorer.initialize()

    # With ignore_top=False (default), uncompressed top layers mismatch shape [512] vs [384] -> raises RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        scorer.load_model(uncompressed_model_dir, ignore_top=False)
    assert "size mismatch" in str(excinfo.value)

    # With ignore_top=True, top layers are discarded -> succeeds cleanly
    scorer.load_model(uncompressed_model_dir, ignore_top=True)
    assert scorer._initialized


def test_load_model_preserves_device_and_buffers_on_initialized_model():
    """Verify that load_model on an already-initialized model updates weights in-place,
    preserving parameter tensor objects, parameter device, and buffer devices (preventing DDP CPU buffer sync crashes)."""
    import tempfile
    import torch
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer

    scorer_cfg, _ = mice_v2_scorer(
        hf_id="hf-internal-testing/tiny-random-ModernBertModel",
        n_contextualization_layers=1,
        n_interaction_layers=1,
        utility_pred=True,
    )
    scorer = scorer_cfg.instance()
    scorer.initialize()

    # Determine target device (MPS if on Apple Silicon, CUDA if on GPU, otherwise CPU)
    if torch.cuda.is_available():
        target_device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        target_device = torch.device("mps")
    else:
        target_device = torch.device("cpu")

    scorer.to(target_device)
    initial_p = next(scorer.parameters())
    initial_param_id = id(initial_p)
    assert initial_p.device.type == target_device.type

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        scorer.save_model(tmp_path)

        # Call load_model on the already initialized & placed model
        scorer.load_model(tmp_path)

        # 1. Parameter identities MUST be preserved (in-place weight update)
        reloaded_p = next(scorer.parameters())
        assert id(reloaded_p) == initial_param_id, "load_model re-instantiated module instead of loading state_dict in-place!"
        assert reloaded_p.device.type == target_device.type

        # 2. All registered buffers MUST remain on target_device
        named_bufs = dict(scorer.hf_model.named_buffers())
        assert len(named_bufs) > 0, "Expected registered buffers in MICE model"
        for buf_name, buf in named_bufs.items():
            assert buf.device.type == target_device.type, (
                f"Buffer '{buf_name}' is on device {buf.device}, expected {target_device}!"
            )


def test_validate_and_align_ignores_unpopulated_field_descriptors():
    """Verify that validate_and_align_config never copies class-level `field` descriptors
    from old model checkpoints where newly added parameters were unpopulated."""
    from training_utils import validate_and_align_config
    from MICE.experiments.micev2_training import MiceV2_FineTuning
    from MICE.modeling.mice_v2.modernbert_xpm import mice_v2_scorer
    from experimaestro.core.arguments import field

    cfg = MiceV2_FineTuning()
    scorer_cfg, _ = mice_v2_scorer()
    old_model = scorer_cfg.instance()
    # 1. With field descriptor implemented, unpopulated instance access properly evaluates to its default (None)
    assert getattr(old_model, "doc_bottleneck_dim", None) is None

    validate_and_align_config(cfg, old_model)
    assert cfg.doc_bottleneck_dim is None
    assert not isinstance(cfg.doc_bottleneck_dim, field)

    # 2. Defensive guard check: even if an unpatched environment injects a raw field object, validate_and_align ignores it
    object.__setattr__(old_model, "doc_bottleneck_dim", field(default=None))
    cfg_legacy = MiceV2_FineTuning()
    validate_and_align_config(cfg_legacy, old_model)
    assert cfg_legacy.doc_bottleneck_dim is None
    assert not isinstance(cfg_legacy.doc_bottleneck_dim, field)


def test_init_micev2_from_checkpoint_with_freeze_top_layers():
    """Verify that InitMICEV2FromCheckpoint with freeze_top_layers=True freezes top layers and head,
    while leaving bottleneck projection parameters trainable."""
    import torch
    from MICE.modeling.mice_v2.modernbert_xpm import (
        mice_v2_scorer,
        InitMICEV2FromCheckpoint,
    )

    loader = resolve_trained_model_loader("VictorMorand/MiceV2-l10p4-ettin-68m")
    source_model = loader.model

    scorer_cfg, _ = mice_v2_scorer(
        hf_id=source_model.hf_id,
        n_contextualization_layers=source_model.n_contextualization_layers,
        n_interaction_layers=source_model.n_interaction_layers,
        bound_bottom_layers=source_model.bound_bottom_layers,
        use_upper_rope=source_model.use_upper_rope,
        top_layers_doc_ln=source_model.top_layers_doc_ln,
        bottom_final_norm=source_model.bottom_final_norm,
        top_final_norm=source_model.top_final_norm,
        compress_dim=1.0,
        compress_init_option="seed",
        doc_bottleneck_dim=256,
        utility_pred=True,
    )

    init_task = InitMICEV2FromCheckpoint.C(
        scorer=scorer_cfg,
        checkpoint_loader=loader,
        freeze_bottom_layers=True,
        freeze_top_layers=True,
    ).instance()

    init_task.execute()

    scorer = init_task.scorer
    assert scorer._initialized
    hf_model = scorer.hf_model

    # Bottom layers: frozen
    for param in hf_model.model.layers.parameters():
        assert not param.requires_grad
    for param in hf_model.model.embeddings.parameters():
        assert not param.requires_grad

    # Top layers and classifier: frozen
    for param in hf_model.top_layers.parameters():
        assert not param.requires_grad
    for param in hf_model.output_modules.parameters():
        assert not param.requires_grad

    # Bottleneck parameters: trainable
    assert hf_model.doc_compress is not None
    assert hf_model.doc_compress.weight.requires_grad
    assert hf_model.doc_decompress is not None
    assert hf_model.doc_decompress.weight.requires_grad


def test_build_mice_v2_with_freeze_top_layers():
    """Verify that build_mice_v2 propagates freeze_top_layers=True to InitMICEV2FromCheckpoint."""
    from MICE.experiments.micev2_training import MiceV2_FineTuning, build_mice_v2, InitMICEV2FromCheckpoint

    cfg = MiceV2_FineTuning()
    cfg.from_experiment = "VictorMorand/MiceV2-l10p4-ettin-68m"
    cfg.freeze_top_layers = True
    cfg.doc_bottleneck_dim = 512

    scorer, init_tasks = build_mice_v2(cfg)

    assert len(init_tasks) == 1
    init_task = init_tasks[0]
    assert isinstance(init_task, InitMICEV2FromCheckpoint)
    assert init_task.freeze_top_layers is True


def test_get_name_from_tags_freeze_top():
    """Verify that get_name_from_tags appends -freezetop when freeze_top_layers is True."""
    from MICE.experiments.micev2_training import MiceV2_FineTuning, get_name_from_tags

    cfg = MiceV2_FineTuning()
    cfg.base = "jhu-clsp/ettin-encoder-17m"
    cfg.n_contextualization_layers = 10
    cfg.freeze_bottom_layers_phase1 = True
    cfg.freeze_top_layers = True
    cfg.doc_bottleneck_dim = 512

    model_tags = {
        "base": cfg.base,
        "n_contextualization_layers": 10,
        "freeze_bottom_layers_phase1": True,
        "freeze_top_layers": True,
        "doc_bottleneck_dim": 512,
    }

    name = get_name_from_tags(model_tags, cfg)
    assert "-freezebottom" in name
    assert "-freezetop" in name
    assert "-dbot512" in name


def test_parse_experiment_ref(monkeypatch, tmp_path):
    """Test parse_experiment_ref correctly distinguishes local runs from HF Hub repos."""
    # HF Hub ID with slash
    exp_id, run_id = parse_experiment_ref("VictorMorand/MiceV2-l10p4-ettin-68m")
    assert exp_id == "VictorMorand/MiceV2-l10p4-ettin-68m"
    assert run_id is None

    # Single string without slash
    exp_id, run_id = parse_experiment_ref("bert-base-uncased")
    assert exp_id == "bert-base-uncased"
    assert run_id is None

    # Local experiment with run ID
    fake_exp_dir = tmp_path / "mock_local_exp"
    fake_exp_dir.mkdir()
    monkeypatch.setattr(
        "training_utils.find_candidate_experiment_dirs",
        lambda eid: [fake_exp_dir] if eid == "mock_local_exp" else [],
    )

    exp_id, run_id = parse_experiment_ref("mock_local_exp/20260924_180227")
    assert exp_id == "mock_local_exp"
    assert run_id == "20260924_180227"


def test_find_experiment_results_model_dir_with_pinned_run(monkeypatch, tmp_path):
    """Test resolving a specific pinned run ID (exp_id/run_id)."""
    exp_dir = tmp_path / "test_exp"
    run1 = exp_dir / "20260924_180227" / "results" / "models" / "ModelA"
    run2 = exp_dir / "20260925_120000" / "results" / "models" / "ModelB"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)
    (run1 / "experimaestro.json").write_text("{}")
    (run2 / "experimaestro.json").write_text("{}")

    # current points to run2
    (exp_dir / "current").symlink_to(exp_dir / "20260925_120000")

    monkeypatch.setattr(
        "training_utils.find_candidate_experiment_dirs",
        lambda eid: [exp_dir] if eid == "test_exp" else [],
    )

    # 1. Pinned to run1
    resolved_run1 = find_experiment_results_model_dir("test_exp/20260924_180227")
    assert resolved_run1 == run1

    # 2. Default (omitted run_id) resolves to current (run2)
    resolved_current = find_experiment_results_model_dir("test_exp")
    assert resolved_current == run2

    # 3. Non-existent run_id raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Run ID 'non_existent_run' not found"):
        find_experiment_results_model_dir("test_exp/non_existent_run")


def test_find_experiment_results_model_dir_fallback_when_current_in_progress(monkeypatch, tmp_path):
    """Test that when 'current' run is in progress (no models yet), it falls back to the latest completed run."""
    exp_dir = tmp_path / "test_exp"
    completed_run = exp_dir / "20260924_180227" / "results" / "models" / "ModelCompleted"
    completed_run.mkdir(parents=True)
    (completed_run / "experimaestro.json").write_text("{}")

    # In-progress run has no results/models directory
    in_progress_run = exp_dir / "20260929_180000"
    in_progress_run.mkdir(parents=True)

    # current symlink points to in_progress_run
    (exp_dir / "current").symlink_to(in_progress_run)

    monkeypatch.setattr(
        "training_utils.find_candidate_experiment_dirs",
        lambda eid: [exp_dir] if eid == "test_exp" else [],
    )

    # Resolving without run_id should automatically fall back to completed_run
    resolved = find_experiment_results_model_dir("test_exp")
    assert resolved == completed_run


def test_find_experiment_results_model_dir_no_completed_runs_raises(monkeypatch, tmp_path):
    """Test that when no completed runs have exported models, an informative FileNotFoundError is raised."""
    exp_dir = tmp_path / "test_exp"
    in_progress_run = exp_dir / "20260929_180000"
    in_progress_run.mkdir(parents=True)
    (exp_dir / "current").symlink_to(in_progress_run)

    monkeypatch.setattr(
        "training_utils.find_candidate_experiment_dirs",
        lambda eid: [exp_dir] if eid == "test_exp" else [],
    )

    with pytest.raises(FileNotFoundError, match="no completed runs with exported models"):
        find_experiment_results_model_dir("test_exp")


def test_resolve_trained_model_loader_local_does_not_swallow_error(monkeypatch, tmp_path):
    """Test that if a local experiment has no valid models, resolve_trained_model_loader raises
    FileNotFoundError and does NOT fall back to Hugging Face Hub."""
    exp_dir = tmp_path / "local_exp"
    exp_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "training_utils.find_candidate_experiment_dirs",
        lambda eid: [exp_dir] if eid == "local_exp" else [],
    )

    with pytest.raises(FileNotFoundError, match="no completed runs with exported models"):
        resolve_trained_model_loader("local_exp")


def test_validate_and_align_config_pinned_run_tags(monkeypatch, tmp_path):
    """Test that validate_and_align_config preserves clean experiment ID in tags['from_experiment']
    and records tags['from_run_id']."""
    exp_dir = tmp_path / "train_micev2_ettin68m_l10+4_v3"
    exp_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "training_utils.find_candidate_experiment_dirs",
        lambda eid: [exp_dir] if eid == "train_micev2_ettin68m_l10+4_v3" else [],
    )

    cfg = MiceV2_FineTuning(
        from_experiment="train_micev2_ettin68m_l10+4_v3/20260924_180227"
    )
    tags = {}

    from types import SimpleNamespace
    dummy_model = SimpleNamespace(
        hf_id="cross-encoder/ettin-reranker-68m-v1",
        n_contextualization_layers=10,
        n_interaction_layers=4,
    )

    validate_and_align_config(
        cfg,
        dummy_model,
        tags=tags,
        from_ref=cfg.from_experiment,
    )

    assert tags["from_experiment"] == "train_micev2_ettin68m_l10+4_v3"
    assert tags["from_run_id"] == "20260924_180227"
    assert cfg.from_run_id == "20260924_180227"
