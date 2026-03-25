# Maintainability Refactor Plan

This plan describes how to make the project easier to maintain without changing the overall behavior of the current pipeline.

## Current Pain Points

The main maintainability issues today are:

- `train.py` mixes argument parsing, dataset selection, model setup, training loops, evaluation, logging, scheduling, and checkpointing in one file.
- Dataset-specific logic is repeated across `train.py`, `run_phase2.py`, `run_phase3.py`, and `run_phase4.py`.
- Shared concepts like `num_classes`, `is_32x32`, and dataset paths are inferred in multiple places.
- The phase scripts are thin enough to keep, but they still duplicate dataset/model bootstrapping logic.
- The code is usable, but it is getting harder to debug and extend safely.

## Refactor Goal

Keep the project script-based, but move reusable logic into small modules so that:

- `train.py` becomes a thin entrypoint
- each phase script only contains phase-specific behavior
- dataset/model setup is centralized
- training/evaluation loops live in one place
- Colab notebooks can call stable Python APIs instead of re-implementing logic

## Recommended Target Structure

```text
src/
  attacks/
  data/
    datasets.py
    registry.py
  detector/
  models/
  training/
    engine.py
    metrics.py
    checkpointing.py
    logging_utils.py
  pipeline/
    bootstrap.py
    config.py
```

## Refactor Phases

### Phase 1: Centralize dataset metadata

Create `src/data/registry.py`.

This module should expose one canonical source of truth for:

- dataset name
- loader function
- root path handling
- number of classes
- whether the model should use `is_32x32`
- recommended batch size defaults
- whether the task is multiclass or multilabel

Example responsibility:

```python
DATASET_CONFIGS = {
    'cifar100': {...},
    'gtsrb': {...},
    'vggface': {...},
    'chexpert': {...},
}
```

This removes repeated `if args.dataset == ...` blocks from multiple scripts.

### Phase 2: Move model/data bootstrap into one shared helper

Create `src/pipeline/bootstrap.py`.

This module should provide reusable helpers like:

- `build_dataloaders(dataset_name, data_dir, batch_size)`
- `build_model(dataset_name, device)`
- `resolve_checkpoint_path(out_dir, dataset_name)`

Once this exists, `train.py`, `run_phase2.py`, `run_phase3.py`, and `run_phase4.py` can all call the same bootstrap functions instead of each script maintaining its own dataset setup logic.

### Phase 3: Extract training loop code

Move the reusable training logic from `train.py` into `src/training/engine.py`.

Suggested functions:

- `train_epoch(...)`
- `evaluate(...)`
- `fit(...)`
- `final_test(...)`

This makes the training behavior testable and keeps `train.py` focused on parsing args and calling the engine.

### Phase 4: Extract metrics and logging

Create:

- `src/training/metrics.py`
- `src/training/logging_utils.py`
- `src/training/checkpointing.py`

Responsibilities:

- metric calculations
- TensorBoard logging helpers
- save/load best checkpoint helpers
- standardized run summaries

This will reduce clutter in the training script and make it easier to add future metrics like macro-F1, multilabel accuracy, and corruption counters.

### Phase 5: Normalize config handling

Create `src/pipeline/config.py`.

Use either:

- a small dataclass-based config object, or
- plain dictionaries that are normalized in one place

The main goal is that every script works off the same resolved config values.

That means things like:

- dataset root directory
- checkpoint filenames
- result filenames
- model shape expectations
- default poison counts

are defined once.

### Phase 6: Keep scripts thin

After the shared modules exist, each top-level script should do very little.

#### `train.py`
Should only:

- parse args
- build config
- build dataloaders/model
- call `fit(...)`
- print/save final summary

#### `run_phase2.py`
Should only:

- parse args
- bootstrap dataset/model
- run poison generation and detection

#### `run_phase3.py`
Should only:

- parse args
- bootstrap dataset/model
- run watermark signature extraction

#### `run_phase4.py`
Should only:

- parse args
- bootstrap dataset/model
- simulate unauthorized training
- generate report

## What should not change during refactor

The following should stay stable during the first cleanup pass:

- CLI argument names
- output file names
- checkpoint naming conventions
- current dataset behavior
- current phase ordering

This keeps the refactor low risk.

## Recommended implementation order

1. Add `src/data/registry.py`
2. Add `src/pipeline/bootstrap.py`
3. Move `train_epoch` and `evaluate` into `src/training/engine.py`
4. Update `train.py` to call shared code
5. Update the three phase scripts to use shared bootstrap helpers
6. Add optional notebook wrappers that call the scripts or shared APIs

## Expected benefits

After this refactor:

- fewer copy/paste bugs across scripts
- easier debugging of dataset-specific issues
- easier Colab usage
- simpler future support for multilabel CheXpert training
- easier testing of loaders, metrics, and phase logic independently

## Suggested deliverables

A good first refactor milestone would be:

- `src/data/registry.py`
- `src/pipeline/bootstrap.py`
- `src/training/engine.py`
- a simplified `train.py`

That alone would remove most of the current maintenance burden while keeping the project structure familiar.
