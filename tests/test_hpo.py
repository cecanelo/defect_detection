import pytest
import optuna

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.hpo import parse_args, sample_hyperparams


# --- parse_args tests ---

def test_parse_args_required():
    args = parse_args([
        '--config', 'configs/config.yaml',
        '--hpo-config', 'configs/hpo.yaml',
        '--model', 'custom_cnn',
    ])
    assert args.config == 'configs/config.yaml'
    assert args.hpo_config == 'configs/hpo.yaml'
    assert args.model == 'custom_cnn'


def test_parse_args_missing_required():
    with pytest.raises(SystemExit):
        parse_args(['--config', 'configs/config.yaml'])


# --- sample_hyperparams tests ---

@pytest.fixture
def trial():
    study = optuna.create_study()
    return study.ask()


def test_sample_hyperparams_log_uniform(trial):
    search_space = {'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2}}
    params = sample_hyperparams(trial, search_space)
    assert 1e-4 <= params['learning_rate'] <= 1e-2


def test_sample_hyperparams_uniform(trial):
    search_space = {'dropout': {'type': 'uniform', 'low': 0.2, 'high': 0.7}}
    params = sample_hyperparams(trial, search_space)
    assert 0.2 <= params['dropout'] <= 0.7


def test_sample_hyperparams_categorical(trial):
    search_space = {'batch_size': {'type': 'categorical', 'values': [16, 32, 64]}}
    params = sample_hyperparams(trial, search_space)
    assert params['batch_size'] in [16, 32, 64]


def test_sample_hyperparams_unknown_type(trial):
    search_space = {'bogus': {'type': 'not_a_real_type'}}
    with pytest.raises(ValueError):
        sample_hyperparams(trial, search_space)


def test_sample_hyperparams_returns_all_keys(trial):
    search_space = {
        'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2},
        'batch_size': {'type': 'categorical', 'values': [16, 32, 64]},
    }
    params = sample_hyperparams(trial, search_space)
    assert set(params.keys()) == {'learning_rate', 'batch_size'}
