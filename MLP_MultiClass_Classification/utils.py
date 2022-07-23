from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_transformations(img_size, train, augmentation=False):
    if train and augmentation:  # transformations on train data when augmentation is required
        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.reshape(-1, img_size * img_size * 3)),
        ])
    else:
        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.Lambda(lambda x: x.reshape(-1, img_size * img_size * 3)),
        ])

    return image_transforms


def get_one_hot(label, num_classes):
    label = torch.tensor(label)
    one_hot_encoded_label = F.one_hot(label, num_classes)
    return one_hot_encoded_label


def visualize_samples(dataset, n_samples, cols=4):
    rows = n_samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))

    for row in range(rows):
        for col in range(cols):
            index = row + col
            sample = dataset[index]  # fetching one sample from dataset
            image = sample['image']
            image = image.reshape(3, 64, 64)
            image = image.numpy().transpose(1, 2, 0)
            label = sample['label']
            ax[row, col].imshow(image)
            ax[row, col].set_title(label)
    plt.show()


def init_weights(net: nn.Module, init_type='uniform'):
    valid_initializations = ['zero_constant', 'uniform']
    classname = net.__class__.__name__
    if init_type not in valid_initializations:
        pass
    elif init_type == valid_initializations[0]:
        if classname.find('Linear') != -1:
            nn.init.constant_(net.weight.data, 0)
            nn.init.constant_(net.bias.data, 0)
    else:
        if classname.find('Linear') != -1:
            net.weight.data.uniform_(-1, 0)
            net.bias.data.uniform_(-1, 0)
