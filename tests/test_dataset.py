import pytest
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import get_dataloaders

@pytest.fixture(scope='module')
def config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)
    
    
@pytest.fixture(scope='module')
def loaders(config):
    return get_dataloaders(config)


def test_batch_image_shape(loaders):
    train_loader, _ = loaders
    images, _ = next(iter(train_loader))
    assert images.shape[1] == 3
    assert images.shape[2] == 224
    assert images.shape[3] == 224


def test_batch_size(loaders, config):
    train_loader, _ = loaders
    images, _ = next(iter(train_loader))
    assert images.shape[0] == config['training']['batch_size']


def test_label_values(loaders):
    train_loader, _ = loaders
    _, labels = next(iter(train_loader))
    assert set(labels.numpy()).issubset({0, 1})    


def test_test_loader_deterministic(loaders):
    _, test_loader = loaders
    batch1, _ = next(iter(test_loader))
    batch2, _ = next(iter(test_loader))
    assert batch1.equal(batch2)

