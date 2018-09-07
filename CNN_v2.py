import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import configparser
import sys
import time
from time_left import pretty_time_left, pretty_print_time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train( train_loader, device, model, criterion, optimizer, epoch, config, start_time ):
    start_epoch = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_time = time.time()
        print('Epoch [{}], Step [{}/{}], Loss: {:.4f}, time since start {}, time in epoch {}, time remaining {}'
            .format(epoch + 1, i + 1, len(train_loader), loss.item(), pretty_print_time(current_time-start_time), pretty_print_time(current_time-start_epoch), 
                pretty_time_left(start_time, i+1, len(train_loader))))

    if config['output'].getboolean('save during training', False) and ((epoch+1) % config['output'].getint('save every nth epoch', 10) == 0):
        torch.save(model.state_dict(), config['output'].get('filename', 'model')+'_after_epoch_{}.ckpt'.format(epoch))        

if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Hyper parameters
    config = configparser.ConfigParser()
    if len(sys.argv) > 1:
        config.read(sys.argv[1])
    else:
        config.read('default.cfg')

    num_epochs = config['hyperparameter'].getint('epochs', 10)
    num_classes = config['hyperparameter'].getint('classes', 4)
    batch_size = config['hyperparameter'].getint('batch size', 10)
    learning_rate = config['hyperparameter'].getfloat('learning rate', 0.01)

    num_samples = config['files'].getint('samples', 10)
    image_size = config['files'].getint('image size', 299)
    training_set_percentage = config['files'].getint('training set percentage', 80)/100.0
    rotation_angle = config['files'].getint('rotation angle', 180)

    print("Paremeters:\nEpochs:\t\t{num_epochs}\nBatch size:\t{batch_size}\nLearning rate:\t{learning_rate}".format(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate))

    data_transform = transforms.Compose([
            transforms.RandomRotation(rotation_angle),
            transforms.RandomResizedCrop(size=image_size, scale=(0.08,0.5), ratio=(0.8,1.25)),
            transforms.ToTensor(),
        ])

    dataset = torchvision.datasets.ImageFolder(root=config['files'].get('data path', './full'),
                                                    transform=data_transform)
    for ii in range(num_samples-1):
        dataset += torchvision.datasets.ImageFolder(root=config['files'].get('data path', './full'), transform=data_transform)

    training_size = int(training_set_percentage * len(dataset))
    if training_size >= len(dataset) or training_size <= 0:
        training_size = int(0.8 * len(dataset))

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, (training_size, len(dataset)-training_size))

    print('Data sets prepared. {} samples of size {}x{}. Training set {} images, test set {}.'.format(len(dataset), image_size, image_size, len(train_dataset), len(test_dataset)))

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    print('Data loaded. Setting up model.')

    model_ft = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model = model_ft.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('Model set up. Ready to train.')

    # Train the model
    start_time = time.time()
    for epoch in range(num_epochs):
        start_epoch = time.time()
        adjust_learning_rate(optimizer, epoch, learning_rate)

        train( train_loader, device, model, criterion, optimizer, epoch, config, start_time )
        
        current_time = time.time()
        print('Epoch [{}/{}] completed, time since start {}, time this epoch {}, time remaining {}'
            .format(epoch + 1, num_epochs, pretty_print_time(current_time-start_time), pretty_print_time(current_time-start_epoch), 
                pretty_time_left(start_time, epoch+1, num_epochs)))

    # Test the model
    validate()

      # Save the model checkpoint
    torch.save(model.state_dict(), config['output'].get('filename', 'model')+'.ckpt')  

def validate( model, criterion, num_classes, test_loader ):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        losses = AverageMeter()
        confusion = torch.zeros((num_classes,num_classes), dtype=torch.float)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for pred, lab in zip(predicted, labels):
                confusion[pred, lab] += 1

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        if num_samples > 1:
            print('Classes: {}'.format(dataset.datasets[1].class_to_idx))
        else:
            print('Classes: {}'.format(dataset.class_to_idx))
        print('Confusion matrix:\n', (confusion))
    
    return losses, correct/total


