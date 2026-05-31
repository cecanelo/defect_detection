import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import parse_args, apply_overrides, setup_device, train_one_epoch, validate


# --- Fixtures ---

@pytest.fixture(scope='module')
def device():
    return torch.device('cpu')


@pytest.fixture(scope='module')
def tiny_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )


@pytest.fixture(scope='module')
def tiny_loader():
    images = torch.randn(16, 3, 8, 8)
    labels = torch.randint(0, 2, (16,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture(scope='module')
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture(scope='module')
def optimizer(tiny_model):
    return AdamW(tiny_model.parameters(), lr=0.001)


# --- parse_args tests ---

def test_parse_args_defaults():
    args = parse_args(['--config', 'configs/config.yaml'])
    assert args.config == 'configs/config.yaml'
    assert args.device is None


def test_parse_args_with_device():
    args = parse_args(['--config', 'configs/config.yaml', '--device', 'cpu'])
    assert args.device == 'cpu'


def test_parse_args_missing_config():
    with pytest.raises(SystemExit):
        parse_args([])


# --- apply_overrides tests ---

def test_apply_overrides_with_device():
    config = {'training': {}}
    args = parse_args(['--config', 'configs/config.yaml', '--device', 'cpu'])
    config = apply_overrides(config, args)
    assert config['training']['device'] == 'cpu'


def test_apply_overrides_no_device():
    config = {'training': {}}
    args = parse_args(['--config', 'configs/config.yaml'])
    config = apply_overrides(config, args)
    assert 'device' not in config['training']


# --- setup_device tests ---

def test_setup_device_default():
    config = {'training': {}}
    device = setup_device(config)
    assert isinstance(device, torch.device)


def test_setup_device_override():
    config = {'training': {'device': 'cpu'}}
    device = setup_device(config)
    assert str(device) == 'cpu'


# --- train_one_epoch tests ---

def test_train_one_epoch_return_types(tiny_model, tiny_loader, criterion, optimizer, device):
    loss, accuracy = train_one_epoch(tiny_model, tiny_loader, criterion, optimizer, device)
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)


def test_train_one_epoch_ranges(tiny_model, tiny_loader, criterion, optimizer, device):
    loss, accuracy = train_one_epoch(tiny_model, tiny_loader, criterion, optimizer, device)
    assert loss > 0
    assert 0.0 <= accuracy <= 1.0


# --- validate tests ---

def test_validate_return_types(tiny_model, tiny_loader, criterion, device):
    val_loss, accuracy, recall, precision, fbeta = validate(tiny_model, tiny_loader, criterion, device, beta=2)
    assert all(isinstance(v, float) for v in [val_loss, accuracy, recall, precision, fbeta])


def test_validate_ranges(tiny_model, tiny_loader, criterion, device):
    val_loss, accuracy, recall, precision, fbeta = validate(tiny_model, tiny_loader, criterion, device, beta=2)
    assert val_loss > 0
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= fbeta <= 1.0
