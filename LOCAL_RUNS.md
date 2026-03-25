# Local Experiment Runs

This project can now be run locally or on a remote server with the same entrypoints.

## Quick local smoke test with real CheXpert data

```bash
python3 run_experiment.py --config configs/local_smoke_chexpert.json
```

## Dry-run a local command without starting training

```bash
python3 run_experiment.py --config configs/local_smoke_chexpert.json --dry-run
```

## Override config values from the command line

```bash
python3 run_experiment.py \
  --config configs/local_smoke_chexpert.json \
  --epochs 2 \
  --batch-size 4 \
  --device cpu \
  --num-workers 0
```

## Train only on local real data with small settings

```bash
python3 run_experiment.py \
  --dataset chexpert \
  --stages train \
  --data-dir ./data \
  --checkpoint-dir ./checkpoints/local_chexpert_train \
  --results-dir ./results/local_chexpert_train \
  --batch-size 8 \
  --epochs 1 \
  --lr 0.01 \
  --num-workers 0 \
  --device auto
```

## Run a tiny local train + phase 2 experiment

```bash
python3 run_experiment.py \
  --dataset chexpert \
  --stages train phase2 \
  --data-dir ./data \
  --checkpoint-dir ./checkpoints/local_chexpert_phase2 \
  --results-dir ./results/local_chexpert_phase2 \
  --batch-size 8 \
  --epochs 1 \
  --num-poisons 4 \
  --poison-steps 5 \
  --clean-subset-size 128 \
  --num-workers 0 \
  --device auto
```

## Run a tiny local end-to-end experiment

```bash
python3 run_experiment.py \
  --dataset cifar100 \
  --stages train phase2 phase3 phase4 \
  --data-dir ./data \
  --checkpoint-dir ./checkpoints/local_cifar100_full \
  --results-dir ./results/local_cifar100_full \
  --batch-size 32 \
  --epochs 1 \
  --phase4-epochs 1 \
  --num-poisons 8 \
  --poison-steps 5 \
  --clean-subset-size 256 \
  --num-workers 0 \
  --device auto
```

## Direct script control

You can still run the original scripts directly:

```bash
python3 train.py --dataset chexpert --epochs 1 --batch-size 8 --device cpu --num-workers 0
python3 run_phase2.py --dataset chexpert --model-path ./checkpoints/local_chexpert_train/best_chexpert_resnet18.pth --num-poisons 4 --poison-steps 5 --clean-subset-size 128 --device cpu --num-workers 0
python3 run_phase3.py --dataset chexpert --model-path ./checkpoints/local_chexpert_train/best_chexpert_resnet18.pth --num-poisons 4 --poison-steps 5 --device cpu --num-workers 0
python3 run_phase4.py --dataset chexpert --auth-model-path ./checkpoints/local_chexpert_train/best_chexpert_resnet18.pth --epochs 1 --num-poisons 4 --poison-steps 5 --device cpu --num-workers 0
```

## Practical local tips

- Use `--device cpu` and `--num-workers 0` for the easiest debugging path.
- Use `--dry-run` first when chaining multiple stages.
- Keep `--epochs`, `--phase4-epochs`, `--num-poisons`, `--poison-steps`, and `--clean-subset-size` small for real-data smoke tests.
- Separate `checkpoint-dir` and `results-dir` per experiment so local tests do not overwrite larger runs.
