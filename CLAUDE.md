# Casting Defect Classifier

Binary CNN classifier for detecting defective vs ok cast pump impeller images.
Dataset: Kaggle "casting product image data for quality inspection" by ravirajsinh45.
300x300 grayscale images, resized to 224x224 in the pipeline.

## Project Structure

```
data/               # Dataset (gitignored) - train/def_front, train/ok_front, test/def_front, test/ok_front
src/
  dataset.py        # get_dataloaders(config) - transforms, ImageFolder, DataLoaders
  model.py          # get_model(config) - CustomCNN and ResNet18 factory
  train.py          # Orchestrator - loads config, runs training and validation loops, MLflow logging
  evaluate.py       # Post-training evaluation - confusion matrix, precision, recall, F1
configs/
  config.yaml       # Single source of truth for all hyperparameters and paths
experiments/        # MLflow tracking and model checkpoints (gitignored)
notebooks/          # Exploratory work
tests/
  test_dataset.py   # pytest tests for dataset pipeline
conftest.py         # pytest root anchor
Dockerfile          # Containerized training pipeline
```

## Environment

```bash
source venv/bin/activate
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Training (not yet implemented)

```bash
python src/train.py --config configs/config.yaml
```

## Key Design Decisions

- **Config-driven:** all hyperparameters and paths live in config.yaml, nothing hardcoded
- **CLI overrides:** training script accepts argument overrides for switching environments (e.g. local vs Colab)
- **Device detection:** auto-detected via torch.cuda.is_available() by default; overridden only if --device CLI argument is passed
- **Image size:** resized to 224x224 inside the pipeline for ResNet18 compatibility and fast iteration
- **Normalization:** ImageNet mean/std used for both CustomCNN and ResNet18 to keep the data pipeline identical regardless of model
- **Augmentations:** train only - random horizontal flip, vertical flip, rotation (15 degrees)
- **Transfer learning ready:** get_model(config) routes to CustomCNN or ResNet18 based on model.name in config

## Planned Progression

1. Local setup -- done
2. Data pipeline -- done
3. Baseline CNN -- in progress
4. Transfer learning -- ResNet18, feature extraction then fine-tuning
5. MLflow integration
6. Optuna HPO
7. Colab + GPU
8. Evaluation -- prioritize recall on defective class
9. Docker

## Dataset Notes

- Train: 3,758 defective, 2,875 ok (defective is majority class)
- Test: 453 defective, 262 ok
- Class imbalance may require loss weighting -- to be addressed in training loop
- ImageFolder class labels: def_front=0, ok_front=1 (assigned alphabetically)
