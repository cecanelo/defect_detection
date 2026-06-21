import torch
import torch.nn as nn
from torchvision import models

class CustomCNN(nn.Module):

    def __init__(self, num_classes, dropout, img_size, hidden_size):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )

        n_features = self._get_n_features(img_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )      
    
    def _get_n_features(self, img_size):
        dummy = torch.zeros(1, 3, img_size, img_size)
        output = self.features(dummy)
        return output.numel()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet18(num_classes, feature_extract=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    
    model.fc = nn.Linear(512, num_classes)

    return model

def get_model(config):
    num_classes = config['model']['num_classes']
    dropout     = config['model']['dropout']
    model_name  = config['model']['name']
    img_size    = config['data']['img_size']
    hidden_size = config['model']['hidden_size']
    feature_extract = config['model'].get('feature_extract', True)

    if model_name == 'custom_cnn':
        return CustomCNN(num_classes, dropout, img_size, hidden_size)
    elif model_name == 'resnet18':
        return get_resnet18(num_classes, feature_extract)
    else:
        raise ValueError(f'Unknown model name: {model_name}')