from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# imagenet_mean and imagenet_std as constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_dataloaders(config):
    train_dir = config['data']['train_dir']
    test_dir = config['data']['test_dir']
    img_size = config['data']['img_size']
    num_workers = config['data']['num_workers']
    batch_size = config['training']['batch_size']
    h_flip = config['augmentation']['random_horizontal_flip']
    v_flip = config['augmentation']['random_vertical_flip']
    rotation = config['augmentation']['random_rotation']

    train_transform_list = [transforms.Grayscale(num_output_channels=3),
                            transforms.Resize((img_size, img_size))
                            ]
    if h_flip:
        train_transform_list.append(transforms.RandomHorizontalFlip())
    if v_flip:
        train_transform_list.append(transforms.RandomVerticalFlip())
    if rotation:
        train_transform_list.append(transforms.RandomRotation(degrees=rotation))
    train_transform_list.extend([transforms.ToTensor(),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                                ])
    train_transforms = transforms.Compose(train_transform_list)

    test_transform_list = [transforms.Grayscale(num_output_channels=3),
                           transforms.Resize((img_size, img_size))
                           ]
    test_transform_list.extend([transforms.ToTensor(),
                                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                                ])
    test_transforms = transforms.Compose(test_transform_list)

    train_dataset = ImageFolder(train_dir, train_transforms)
    test_dataset = ImageFolder(test_dir, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader