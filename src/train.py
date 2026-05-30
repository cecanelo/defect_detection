import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
import mlflow
from sklearn.metrics import recall_score, fbeta_score, accuracy_score

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.device is not None:
        config['training']['device'] = args.device
        logger.info(f'Device overriden via CLI: {args.device}')
    return config


def setup_device(config: dict, override: str = None) -> torch.device:
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
) -> tuple[float, float, float, float]:

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

    return avg_loss, accuracy, recall, fbeta
