from __future__ import print_function
import argparse
import socket
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet18_Weights, ResNet101_Weights
from torchsummary import summary

from grid_builder.LabelBuilder import LabelBuilder
from models_pytorch.dataset import DataHelper, ImageGeolocationDataset
from models_pytorch.utils import create_datahelper, get_model, get_preprocessing


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, epoch, duration):
    model.eval()
    test_loss = 0
    correct = 0


    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest Set Epoch: {} Duration: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, datetime.timedelta(seconds=duration), test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch Geolocation classifier')

    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help='pretrained model, supported  values: resnet18, resnet34, resnet50, resnet101 (default: resnet18)')
    parser.add_argument('--augmentations', action='store_true', default=False,
                        help='For training data augmentations')
    parser.add_argument('--dataset', type=str, default='flickr_images', metavar='N',
                        help='dataset, supported are geotags_185K or flickr_images (default: flickr_images)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=1, metavar='M',
                        help='Learning rate step size (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'Commandline args: {args}')
    print(f'Device: {device}')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    labelBuilder = LabelBuilder()
    data_helper = create_datahelper(args.dataset, args.seed)
    preprocessing = get_preprocessing(args.model)
    training_dataset = ImageGeolocationDataset(data_helper.training_data, preprocessing=preprocessing, augumentation=args.augmentations)
    test_dataset = ImageGeolocationDataset(data_helper.test_data,  preprocessing=preprocessing, augumentation=False)

    num_classes = labelBuilder.get_num_labels()
    data, label = training_dataset[0]
    input_shape = data.shape

    print(f'Input shape {input_shape} output shape {(num_classes,)}')

    train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = get_model(args.model, device, num_classes, input_shape)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    start = time.time()
    model_file_name = f"geolocation_cnn_{args.dataset}_{args.model}.pt"

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch} learning-rate {scheduler.get_last_lr()}')
        train(args, model, criterion, device, train_loader, optimizer, epoch)
        duration = time.time() - start
        test(model, device, test_loader, epoch, duration)
        scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), model_file_name)
            print(f'Saved model in file {model_file_name}')


if __name__ == '__main__':
    main()
