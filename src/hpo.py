import os
import json
import copy
import time
import datetime
import argparse
import logging

import torch

import mlflow
import optuna
from optuna.pruners import MedianPruner

from src.utils import load_config
from src.train import train_one_epoch, validate, setup_device, setup_components


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_OVERRIDES = {
    'custom_cnn': {'name': 'custom_cnn'},
    'resnet18_head': {'name': 'resnet18', 'feature_extract': True},
}


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--hpo-config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    return parser.parse_args(args)


def sample_hyperparams(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}
    for name, spec in search_space.items():
        param_type = spec['type']

        if param_type == 'log_uniform':
            params[name] = trial.suggest_float(name, spec['low'], spec['high'], log=True)
        elif param_type == 'uniform':
            params[name] = trial.suggest_float(name, spec['low'], spec['high'])
        elif param_type == 'categorical':
            params[name] = trial.suggest_categorical(name, spec['values'])
        else:
            raise ValueError(f'Unknown search space type: {param_type}')

    return params


def run_trial(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    beta: float,
    epochs_per_trial: int,
    trial: optuna.Trial,
) -> float:

    best_fbeta = 0.0

    for epoch in range(epochs_per_trial):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, recall, precision, fbeta = validate(model, test_loader, criterion, device, beta)

        scheduler.step()

        mlflow.log_metrics({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'recall': recall,
            'precision': precision,
            'fbeta': fbeta,
            'lr': optimizer.param_groups[0]['lr'],
        }, step=epoch)

        logger.info(
            f'Trial {trial.number} | Epoch {epoch+1}/{epochs_per_trial} | '
            f'train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | '
            f'recall: {recall:.4f} | precision: {precision:.4f} | fbeta: {fbeta:.4f}'
        )

        best_fbeta = max(best_fbeta, fbeta)

        trial.report(fbeta, epoch)
        if trial.should_prune():
            logger.info(f'Trial {trial.number} pruned at epoch {epoch+1}')
            raise optuna.TrialPruned()

    return best_fbeta


def objective(trial: optuna.Trial, config: dict, search_space: dict, model_name: str, device: torch.device) -> float:
    params = sample_hyperparams(trial, search_space)

    trial_config = copy.deepcopy(config)
    trial_config['model'].update(MODEL_OVERRIDES[model_name])
    trial_config['training']['epochs'] = trial_config['hpo']['epochs_per_trial']

    for name, value in params.items():
        if name in trial_config['training']:
            trial_config['training'][name] = value
        elif name in trial_config['model']:
            trial_config['model'][name] = value
        else:
            raise ValueError(f'Unknown hyperparameter: {name}')

    model, train_loader, test_loader, criterion, optimizer, scheduler = setup_components(trial_config, device)

    beta = trial_config['training']['beta']
    epochs_per_trial = trial_config['hpo']['epochs_per_trial']

    with mlflow.start_run(run_name=f'trial_{trial.number}', nested=True):
        mlflow.log_params(params)
        best_fbeta = run_trial(
            model, train_loader, test_loader, criterion, optimizer, scheduler,
            device, beta, epochs_per_trial, trial
        )

    return best_fbeta


def main():
    start_time = time.time()
    args = parse_args()

    config = load_config(args.config)
    hpo_config = load_config(args.hpo_config)

    if args.model not in hpo_config:
        raise ValueError(f'No search space defined for model: {args.model}')
    if args.model not in MODEL_OVERRIDES:
        raise ValueError(f'No model config mapping defined for: {args.model}')
    search_space = hpo_config[args.model]

    device = setup_device(config)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['hpo']['experiment_name'])

    pruner = MedianPruner(
        n_startup_trials=config['hpo']['pruner']['n_startup_trials'],
        n_warmup_steps=config['hpo']['pruner']['n_warmup_steps'],
    )
    study = optuna.create_study(direction='maximize', pruner=pruner)

    with mlflow.start_run(run_name=f'hpo_{args.model}') as parent_run:
        study.optimize(
            lambda trial: objective(trial, config, search_space, args.model, device),
            n_trials=config['hpo']['n_trials'],
        )

        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)
        mlflow.log_metric('best_fbeta', best_trial.value)

        parent_run_id = parent_run.info.run_id

    results_dir = config['paths']['hpo_results_dir']
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f'{args.model}.json')

    record = {
        'model': args.model,
        'parent_run_id': parent_run_id,
        'best_fbeta': best_trial.value,
        'best_params': best_trial.params,
        'timestamp': datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
    }
    with open(json_path, 'w') as f:
        json.dump(record, f, indent=2)

    elapsed = int(time.time() - start_time)
    logger.info(f'HPO complete in {str(datetime.timedelta(seconds=elapsed))}')
    logger.info(f'Best trial : {best_trial.number}')
    logger.info(f'Best F-beta: {best_trial.value:.4f}')
    logger.info(f'Best params: {best_trial.params}')
    logger.info(f'Saved      : {json_path}')


if __name__ == '__main__':
    main()
