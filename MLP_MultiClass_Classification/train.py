import os
import argparse
import yaml

import torch.optim
from dataset import SignDigitDataset
from torch.utils.data import DataLoader
from utils import *
from model import MLP
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/sign_digits_experiment_1')

parser = argparse.ArgumentParser()
# Hyper-parameters
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs for training')
parser.add_argument('--print_every', type=int, default=10, help='print the loss every n epochs')
parser.add_argument('--img_size', type=int, default=64, help='image input size')
parser.add_argument('--n_classes', type=int, default=6, help='number of classes')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_layers', type=int, default=[50, 25], nargs='+',
                    help='number of units per layer (except input and output layer)')
parser.add_argument('--activation', type=str, default=None , choices=['relu', 'tanh', 'sigmoid'],
                    help='activation layers')
parser.add_argument('--initialization', type=str, default=None, choices=['zero_constant', 'uniform'],
                    help='type of initialization')
parser.add_argument('--augmentation', type=bool, default=False, choices=[True, False],
                    help='augmentation requirement')


args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: You are not using gpu!")

# 0. creating train_dataset and test_dataset
train_dataset = SignDigitDataset(root_dir='data/',
                                 h5_name='train_signs.h5',
                                 train=True,
                                 transform=get_transformations(img_size=64, train=True, augmentation=args.augmentation))

test_dataset = SignDigitDataset(root_dir='data/',
                                h5_name='test_signs.h5',
                                train=False,
                                transform=get_transformations(img_size=64, train=False))

# 1. Data loaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

# 2. get an instance of the model
units = [args.img_size * args.img_size * 3]
for hidden_layer in args.hidden_layers:
    units.append(hidden_layer)
units.append(args.n_classes)

model = MLP(units=units, hidden_layer_activation=args.activation, init_type=args.initialization).to(device)
model.apply(lambda net: init_weights(net, model.init_type))
print(model)
# 3, 4. loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

# 5. Train the model
for epoch in range(args.n_epochs):
    train_running_loss, test_running_loss = 0.0, 0.0
    n_train_batches = 0
    n_test_batches = 0

    total_train = 0
    correct_train = 0
    for i, batch in enumerate(train_dataloader):
        model.train()
        n_train_batches += 1
        temp = batch['image']
        images = batch['image'].reshape(-1, units[0]).to(device)  # ([8, 12288])
        labels = batch['label'].to(device)  # ([8])

        prediction = model(images)
        batch_loss = criterion(prediction, labels)
        train_running_loss += batch_loss.item()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # calculating number of correctly predicted
        predicted = torch.argmax(torch.nn.Softmax(dim=1)(prediction), dim=1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.shape[0]

    writer.add_scalar('Loss/train', round(train_running_loss / n_train_batches, 4), epoch)
    writer.add_scalar('Accuracy/train', round(100 * correct_train / total_train, 4), epoch)

    with torch.no_grad():
        correct_test = 0
        total_test = 0
        for i, batch in enumerate(test_dataloader):
            n_test_batches += 1
            images = batch['image'].reshape(-1, units[0]).to(device)
            labels = batch['label'].to(device)

            # calculating loss value on test images
            prediction = model(images)
            batch_loss = criterion(prediction, labels)
            test_running_loss += batch_loss.item()

            # calculating number of correctly predicted
            predicted = torch.argmax(torch.nn.Softmax(dim=1)(prediction), dim=1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.shape[0]

        writer.add_scalar('Loss/test',round( test_running_loss / n_test_batches, 4), epoch)
        writer.add_scalar('Accuracy/test', round(100 * correct_test / total_test, 4), epoch)

    if epoch % args.print_every == 0:
        print(
            'Epoch [{}/{}]:\t Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy:{:.4f}'.format(
                epoch + 1,
                args.n_epochs,
                train_running_loss / n_train_batches,
                100 * correct_train / total_train,
                test_running_loss / n_test_batches,
                100 * correct_test / total_test))

  
# save the model weights
checkpoint_dir = 'checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
path = os.path.join(checkpoint_dir, 'best_model.pth')
torch.save(model.state_dict(), path)

#save model hyperparameters
hyper_parameters = [
    {
        'n_epochs': args.n_epochs,
        'print_every': args.print_every,
        'img_size': args.img_size,
        'hidden_layers': args.hidden_layers,
        'n_classes': args.n_classes,
        'activation': args.activation,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'initialization': args.initialization,
        'augmentation': args.augmentation,
    }
]
with open("params.yaml", 'w') as f:
    data = yaml.dump(hyper_parameters, f)
    print('write successful')

