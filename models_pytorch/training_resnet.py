import copy
import time

import torch
from PIL import Image
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, models

from models_pytorch.dataset import DataHelper, ImageGeolocationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(training_data, test_data, model, criterion, optimizer, scheduler, batch_size=100, num_epochs=25):

    print(f'Starting training model: batch-size: {batch_size} num-epochs: {num_epochs} num-training-data: {len(training_data)} num-test-data: {len(test_data)} device: {device}')
    since = time.time()

    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        start_epoch = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                dataloader = training_data_loader
                dataset_size = len(training_data)
                model.train()  # Set model to training mode
            else:
                dataloader = test_data_loader
                dataset_size = len(test_data)
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch in dataloader:
                inputs = batch[0]
                labels = batch[1]
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Finished epoch, duration {int(time.time() - start_epoch)} seconds')
        print('-' * 10)

        print()

    time_elapsed = time.time() - since
    print()
    print()
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f}')
    print('-' * 20)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main(data_helper):

    trainimg_data_loader = ImageGeolocationDataset(data_helper.training_data)
    test_data_loader = ImageGeolocationDataset(data_helper.test_data)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    # set numer of classes
    model_ft.fc = nn.Linear(num_ftrs, len(data_helper.all_labels))

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(training_data=trainimg_data_loader, test_data=test_data_loader, model=model_ft,
                           criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
                           batch_size=100, num_epochs=25)




if __name__ == '__main__':

    # adnwsrtx01
    base_dir = '/home/test-dev/projects/adncuba-geolocation-classifier/grid_builder'
    data_dir = '/mnt/store/geolocation_classifier/'

    # hacke vmware
    base_dir = '/home/hacke/projects/adncuba-geolocation-classifier/grid_builder'
    data_dir = '/home/hacke/projects/data/geolocation_classifier'

    data_helper_flicker = DataHelper(base_dir=base_dir, dataset_name='flickr_images', data_dir=data_dir, test_fraction=0.8, seed=42)
    main(data_helper_flicker)