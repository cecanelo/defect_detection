# Design Decisions

## Data Pipeline

### Resize inside pipeline, not on disk
Images are resized from 300x300 to 224x224 via torchvision transforms at load time because it is non-destructive and keeps image size a tunable config parameter.

### Grayscale images duplicated to 3 channels
`transforms.Grayscale(num_output_channels=3)` is applied in both pipelines because ResNet18 pretrained weights expect 3-channel input and it keeps the pipeline identical for both models.

### ImageNet normalization for both models
ImageNet mean/std normalization is used for both CustomCNN and ResNet18 because it is required for transfer learning and simplifies the pipeline by avoiding model-specific preprocessing.

### Augmentations on training set only
Flip and rotation augmentations are applied to the training set only because the test set must reflect real-world unmodified images. Augmenting it would distort performance estimates.

---

## Configuration

### Config-driven design
All hyperparameters and paths live in `configs/config.yaml` because it provides a single source of truth and enables clean Optuna integration without touching code.

### CLI overrides
`train.py` accepts CLI arguments that override config values because switching from local to Colab requires only passing different arguments, not maintaining parallel config files.

### Device auto-detection with CLI override
Device is detected via `torch.cuda.is_available()` by default and overridable via `--device` because it works correctly in both local and Colab environments without any config changes.

---

## Model Architecture

### Baseline: CustomCNN with 3 conv blocks
Three convolutional blocks with 32, 64, 128 filters because it is a well-established baseline and the doubling pattern reflects that deeper layers need more filters to detect complex patterns.

### 3x3 kernels with padding=1
All conv layers use kernel_size=3 and padding=1 because 3x3 is the industry standard for efficiency and padding=1 preserves spatial dimensions, making MaxPool the sole controller of spatial reduction.

### Global Average Pooling instead of Flatten
The classifier head uses Global Average Pooling (`nn.AdaptiveAvgPool2d(1)`) before the Linear layer because it reduces parameters from ~25.7M to ~127K while preserving feature information, reducing overfitting risk and keeping HPO iteration fast.

### Dynamic feature size calculation
`_get_n_features()` computes the flattened size via a dummy tensor forward pass because hardcoding it breaks silently when img_size or architecture changes.

### Configurable hidden layer size and dropout
`hidden_size` and `dropout` are config parameters because both are Optuna HPO candidates that need to be overridable without touching model.py.

### Transfer learning: feature extraction before fine-tuning
ResNet18 defaults to feature extraction mode with a frozen backbone because it is faster, less prone to overfitting on small datasets, and establishes a performance baseline before committing to full fine-tuning.

### get_model factory function
`train.py` always calls `get_model(config)` and never instantiates models directly because switching models requires only changing `model.name` in config.

---

## Environment and Tooling

### venv over conda
Virtual environment managed with venv because conda inside Docker adds ~400MB overhead and activation complexity, while venv maps cleanly to pip install in Docker.

### pytest with module-scoped fixtures
pytest with `scope='module'` fixtures is used because it is more concise than unittest, uses plain assert statements, and is consistent with the baseball pitch classifier project.

### conftest.py at project root
An empty conftest.py anchors pytest at the project root because pytest 9.x does not reliably execute sys.path.append before collecting modules.

---

## Training

### Loss Function: CrossEntropyLoss (unweighted)
Plain CrossEntropyLoss is used because defective is already the majority class, so no imbalance correction is needed, and the recall preference is expressed at checkpoint selection via F-beta rather than in the loss.

### Optimizer: AdamW
We are using weight decay and vanilla Adam has a known bug where weight decay interacts poorly with adaptive learning rates.

### Checkpointing metric: F-beta (beta=2)
Best checkpoint is saved based on F-beta (beta=2) on the defective class, weighting recall twice as heavily as precision because a missed defect is worse than a rejected good part. Beta is config-driven but excluded from Optuna's search space, since it defines the objective rather than being optimized against it.

