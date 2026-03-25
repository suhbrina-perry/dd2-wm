import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
VALID_STAGES = ['train', 'phase2', 'phase3', 'phase4']
DEFAULT_CONFIG = {
    'dataset': 'cifar100',
    'stages': ['train'],
    'data_dir': './data',
    'checkpoint_dir': './checkpoints/local_experiment',
    'results_dir': './results/local_experiment',
    'model_path': None,
    'device': 'auto',
    'num_workers': 4,
    'batch_size': 128,
    'epochs': 1,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'num_poisons': 10,
    'target_class': 0,
    'poison_class': 1,
    'poison_steps': 10,
    'poison_lr': 0.1,
    'poison_epsilon': 16 / 255,
    'clean_subset_size': 512,
    'phase4_epochs': 1,
    'stolen_lr': 0.01,
    'stolen_weight_decay': 5e-4,
}


def load_config(config_path: Optional[str]) -> dict:
    config = dict(DEFAULT_CONFIG)
    if config_path is None:
        return config

    with open(config_path, 'r', encoding='utf-8') as handle:
        file_config = json.load(handle)

    config.update(file_config)
    return config


def normalize_stages(stages) -> List[str]:
    if isinstance(stages, str):
        stages = [stage.strip() for stage in stages.split(',') if stage.strip()]

    if not isinstance(stages, list) or not stages:
        raise ValueError('stages must be a non-empty list or comma-separated string.')

    invalid = [stage for stage in stages if stage not in VALID_STAGES]
    if invalid:
        raise ValueError(f'Invalid stages: {invalid}. Valid stages are {VALID_STAGES}.')

    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Server-agnostic experiment runner for local or remote use')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'gtsrb', 'vggface', 'chexpert'], default=None)
    parser.add_argument('--stages', nargs='+', default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--num-poisons', type=int, default=None)
    parser.add_argument('--target-class', type=int, default=None)
    parser.add_argument('--poison-class', type=int, default=None)
    parser.add_argument('--poison-steps', type=int, default=None)
    parser.add_argument('--poison-lr', type=float, default=None)
    parser.add_argument('--poison-epsilon', type=float, default=None)
    parser.add_argument('--clean-subset-size', type=int, default=None)
    parser.add_argument('--phase4-epochs', type=int, default=None)
    parser.add_argument('--stolen-lr', type=float, default=None)
    parser.add_argument('--stolen-weight-decay', type=float, default=None)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    merged = dict(config)
    for key, value in vars(args).items():
        if key in {'config', 'dry_run'}:
            continue
        if value is not None:
            merged[key.replace('-', '_')] = value

    merged['stages'] = normalize_stages(merged['stages'])
    return merged


def build_model_path(config: dict) -> str:
    if config.get('model_path'):
        return config['model_path']
    return os.path.join(config['checkpoint_dir'], f"best_{config['dataset']}_resnet18.pth")


def run_command(command: List[str], dry_run: bool) -> None:
    print(' '.join(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_required_paths(config: dict, model_path: str, dry_run: bool) -> None:
    if not dry_run:
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['results_dir'], exist_ok=True)

    if 'train' not in config['stages'] and not dry_run and not os.path.exists(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    config['dry_run'] = args.dry_run
    model_path = build_model_path(config)
    ensure_required_paths(config, model_path, args.dry_run)

    print(json.dumps({**config, 'model_path': model_path}, indent=2))

    python_exec = sys.executable

    if 'train' in config['stages']:
        run_command([
            python_exec,
            'train.py',
            '--dataset', config['dataset'],
            '--batch-size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--lr', str(config['lr']),
            '--weight-decay', str(config['weight_decay']),
            '--data-dir', config['data_dir'],
            '--out-dir', config['checkpoint_dir'],
            '--device', config['device'],
            '--num-workers', str(config['num_workers']),
        ], args.dry_run)

    if 'phase2' in config['stages']:
        run_command([
            python_exec,
            'run_phase2.py',
            '--dataset', config['dataset'],
            '--batch-size', str(config['batch_size']),
            '--num-poisons', str(config['num_poisons']),
            '--target-class', str(config['target_class']),
            '--poison-class', str(config['poison_class']),
            '--poison-steps', str(config['poison_steps']),
            '--poison-lr', str(config['poison_lr']),
            '--poison-epsilon', str(config['poison_epsilon']),
            '--clean-subset-size', str(config['clean_subset_size']),
            '--data-dir', config['data_dir'],
            '--model-path', model_path,
            '--device', config['device'],
            '--num-workers', str(config['num_workers']),
        ], args.dry_run)

    if 'phase3' in config['stages']:
        run_command([
            python_exec,
            'run_phase3.py',
            '--dataset', config['dataset'],
            '--batch-size', str(config['batch_size']),
            '--num-poisons', str(config['num_poisons']),
            '--target-class', str(config['target_class']),
            '--poison-class', str(config['poison_class']),
            '--poison-steps', str(config['poison_steps']),
            '--poison-lr', str(config['poison_lr']),
            '--poison-epsilon', str(config['poison_epsilon']),
            '--data-dir', config['data_dir'],
            '--model-path', model_path,
            '--device', config['device'],
            '--num-workers', str(config['num_workers']),
        ], args.dry_run)

    if 'phase4' in config['stages']:
        run_command([
            python_exec,
            'run_phase4.py',
            '--dataset', config['dataset'],
            '--batch-size', str(config['batch_size']),
            '--epochs', str(config['phase4_epochs']),
            '--num-poisons', str(config['num_poisons']),
            '--target-class', str(config['target_class']),
            '--poison-class', str(config['poison_class']),
            '--poison-steps', str(config['poison_steps']),
            '--poison-lr', str(config['poison_lr']),
            '--poison-epsilon', str(config['poison_epsilon']),
            '--stolen-lr', str(config['stolen_lr']),
            '--stolen-weight-decay', str(config['stolen_weight_decay']),
            '--data-dir', config['data_dir'],
            '--out-dir', config['results_dir'],
            '--auth-model-path', model_path,
            '--device', config['device'],
            '--num-workers', str(config['num_workers']),
        ], args.dry_run)


if __name__ == '__main__':
    main()
