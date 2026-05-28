import pytest
import yaml
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import get_resnet18, get_model


@pytest.fixture(scope='module')
def config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_custom_cnn_output_shape(config):
    model = get_model(config)
    batch_size = config['training']['batch_size']
    img_size   = config['data']['img_size']
    dummy      = torch.zeros(batch_size, 3, img_size, img_size)
    output     = model(dummy)
    assert output.shape == (batch_size, config['model']['num_classes'])


def test_resnet18_output_shape(config):
    config_copy = dict(config)
    config_copy['model'] = dict(config['model'])
    config_copy['model']['name'] = 'resnet18'
    model  = get_model(config_copy)
    dummy  = torch.zeros(1, 3, 224, 224)
    output = model(dummy)
    assert output.shape == (1, config['model']['num_classes'])


def test_resnet18_backbone_frozen(config):
    model = get_resnet18(num_classes=2, feature_extract=True)
    for name, param in model.named_parameters():
        if name not in ('fc.weight', 'fc.bias'):
            assert not param.requires_grad, f"{name} should be frozen"


def test_get_model_invalid_name(config):
    config_copy = dict(config)
    config_copy['model'] = dict(config['model'])
    config_copy['model']['name'] = 'invalid_model'
    with pytest.raises(ValueError):
        get_model(config_copy)
