import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
import mlflow
from sklearn.metrics import recall_score, fbeta_score, accuracy_score, precision_score
import copy
import time
import datetime

from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import load_config

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args(args)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.device is not None:
        config['training']['device'] = args.device
        logger.info(f'Device overriden via CLI: {args.device}')
    return config


def setup_device(config: dict) -> torch.device:
    override = config['training'].get('device')
    if override is not None:
        device = torch.device(override)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    return device


def setup_components(config: dict, device: torch.device) -> tuple:
    train_loader, test_loader = get_dataloaders(config)

    model = get_model(config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    logger.info(f'Model: {config["model"]["name"]}')
    logger.info(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    return model, train_loader, test_loader, criterion, optimizer


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    beta: float
) -> tuple[float, float, float, float, float]:

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss  = total_loss / len(loader)
    accuracy  = accuracy_score(all_labels, all_preds)
    recall    = recall_score(all_labels, all_preds, pos_label=0)
    fbeta     = fbeta_score(all_labels, all_preds, beta=beta, pos_label=0)
    precision = precision_score(all_labels, all_preds, pos_label=0)

    return avg_loss, accuracy, recall, precision, fbeta


def main():
    start_time = time.time()
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)
    device = setup_device(config)

    model, train_loader, test_loader, criterion, optimizer = setup_components(config, device)

    beta       = config['training']['beta']
    epochs     = config['training']['epochs']
    best_fbeta     = 0.0
    best_state     = None
    best_recall    = 0.0
    best_precision = 0.0

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=config['mlflow']['run_name']):

        mlflow.log_params({
            'model'        : config['model']['name'],
            'epochs'       : epochs,
            'batch_size'   : config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay' : config['training']['weight_decay'],
            'dropout'      : config['model']['dropout'],
            'hidden_size'  : config['model']['hidden_size'],
            'img_size'     : config['data']['img_size'],
            'beta'         : beta,
        })

        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, recall, precision, fbeta = validate(model, test_loader, criterion, device, beta)

            mlflow.log_metrics({
                'train_loss' : train_loss,
                'train_acc'  : train_acc,
                'val_loss'   : val_loss,
                'val_acc'    : val_acc,
                'recall'     : recall,
                'precision'  : precision,
                'fbeta'      : fbeta,
            }, step=epoch)

            logger.info(
                f'Epoch {epoch+1}/{epochs} | '
                f'train_loss: {train_loss:.4f} | '
                f'val_loss: {val_loss:.4f} | '
                f'recall: {recall:.4f} | '
                f'precision: {precision:.4f} | '
                f'fbeta: {fbeta:.4f}'
            )

            if fbeta > best_fbeta:
                best_fbeta     = fbeta
                best_recall    = recall
                best_precision = precision
                best_state     = copy.deepcopy(model.state_dict())


        checkpoint_dir  = config['paths']['checkpoint_dir']
        checkpoint_path = os.path.join(checkpoint_dir, f"{config['mlflow']['run_name']}.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(best_state, checkpoint_path)

        mlflow.log_artifact(checkpoint_path)

        elapsed = int(time.time() - start_time)
        logger.info(f'Training complete in {str(datetime.timedelta(seconds=elapsed))}')
        logger.info(f'Best F-beta : {best_fbeta:.4f}')
        logger.info(f'Recall      : {best_recall:.4f}')
        logger.info(f'Precision   : {best_precision:.4f}')
        logger.info(f'Checkpoint  : {checkpoint_path}')


if __name__ == '__main__':
    main()
