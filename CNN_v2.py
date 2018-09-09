import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import configparser
import sys
import time
from time_left import pretty_time_left, pretty_print_time
import numpy as np

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


def train( train_loader, device, model, criterion, optimizer, epoch, start_time ):
    start_epoch = time.time()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        print(np.unique(labels, return_counts=True))
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # evaluate
        losses.update(loss.item(), images.size(0))
        _, predicted = torch.max(outputs.data, 1)
        top1.update((predicted == labels).sum().item(), labels.size(0))
        print('DEBUG:', labels.size(0), (predicted == labels).sum().item(), loss.item(), images.size(0))


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_time = time.time()
        print('Epoch [{}], Step [{}/{}], Loss: {:.4f}, time since start {}, time in epoch {}, epoch remaining {}'
            .format(epoch + 1, i + 1, len(train_loader), loss.item(), pretty_print_time(current_time-start_time), pretty_print_time(current_time-start_epoch), 
                pretty_time_left(start_epoch, i+1, len(train_loader))))

    return losses, top1   

def validate( model, criterion, num_classes, test_loader, device ):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        losses = AverageMeter()
        top1 = AverageMeter()
        confusion = torch.zeros((num_classes,num_classes), dtype=torch.float)

        for images, labels in test_loader:
            print(np.unique(labels, return_counts=True))
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top1.update((predicted == labels).sum().item(), labels.size(0))
            print('DEBUG:', labels.size(0), (predicted == labels).sum().item(), loss.item(), images.size(0))
            for pred, lab in zip(predicted, labels):
                confusion[pred, lab] += 1
        print('Acuracy: ', str(correct/total))
    
    return losses, top1, confusion


def main():

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

    num_samples = config['files'].getint('samples', 1000)
    image_size = config['files'].getint('image size', 299)
    training_set_percentage = config['files'].getint('training set percentage', 80)/100.0
    rotation_angle = config['files'].getint('rotation angle', 180)

    print("Paremeters:\nEpochs:\t\t{num_epochs}\nBatch size:\t{batch_size}\nLearning rate:\t{learning_rate}".format(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate))

    data_transform = transforms.Compose([
            #transforms.RandomRotation(rotation_angle),
            transforms.RandomResizedCrop(size=image_size, scale=(0.08,1.0), ratio=(0.8,1.25)),
            transforms.ToTensor(),
        ])

    dataset = torchvision.datasets.ImageFolder(root=config['files'].get('data path', './full'),
                                                    transform=data_transform)
    classes = dataset.class_to_idx
    class_distribution = np.array(dataset.imgs)[:,1].astype(int)
    weights = np.zeros(len(classes))
    for ii in range(len(classes)):
        weights[ii] = np.sum(class_distribution == ii)/len(class_distribution)
    inverted_weights = (1/weights)/np.sum(1/weights)
    sampling_weights = np.zeros(class_distribution.shape, dtype=np.float)
    for ii in range(len(classes)):
        sampling_weights[class_distribution == ii] = inverted_weights[ii]
    sampling_weights /= sampling_weights.sum()
    print('Weights:', inverted_weights)

    #for ii in range(num_samples-1):
    #    dataset += torchvision.datasets.ImageFolder(root=config['files'].get('data path', './full'), transform=data_transform)

    training_size = int(training_set_percentage * num_samples)
    if training_size >= num_samples or training_size <= 0:
        training_size = int(0.8 * num_samples)

    training_split = int(training_set_percentage * len(dataset))
    if training_split >= len(dataset) or training_split <= 0:
        training_split = int(0.8 * len(dataset))

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, (training_split, len(dataset)-training_split))
    train_sampler = torch.utils.data.WeightedRandomSampler( sampling_weights[train_dataset.indices], training_size, True )
    test_sampler = torch.utils.data.WeightedRandomSampler( sampling_weights[test_dataset.indices], num_samples-training_size, True )
    #train_sampler = None
    #test_sampler = None

    print('Data sets prepared. {} samples of size {}x{}. Training set {} images, test set {}.'.format(num_samples, image_size, image_size, len(train_dataset), len(test_dataset)))

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            sampler=test_sampler)

    print('Data loaded. Setting up model.')

    model_ft = models.resnet18(pretrained=False, num_classes=num_classes)#, aux_logits=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model = model_ft.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('Model set up. Ready to train.')

    # Train the model
    train_loss = torch.zeros(num_epochs, dtype=torch.float)
    train_accuracy = torch.zeros(num_epochs, dtype=torch.float)
    test_loss = torch.zeros(num_epochs, dtype=torch.float)
    test_accuracy = torch.zeros(num_epochs, dtype=torch.float)
    test_confusion = torch.zeros((num_epochs, num_classes, num_classes), dtype=torch.float)
    start_time = time.time()
    for epoch in range(num_epochs):
        start_epoch = time.time()
        adjust_learning_rate(optimizer, epoch, learning_rate)

        losses, top1 = train( train_loader, device, model, criterion, optimizer, epoch, start_time )
        train_loss[epoch] = losses.avg
        train_accuracy[epoch] = top1.avg/batch_size
        
        losses, top1, confusion = validate( model, criterion, num_classes, test_loader, device )
        test_loss[epoch] = losses.avg
        test_accuracy[epoch] = top1.avg/batch_size
        test_confusion[epoch,:,:] = confusion

        print('Test Accuracy of the model on the {} test images: {} %'.format(top1.count, top1.avg))
        print('Classes: {}'.format(classes))
        print('Confusion matrix:\n', (confusion))

        if config['output'].getboolean('save during training', False) and ((epoch+1) % config['output'].getint('save every nth epoch', 10) == 0):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, config['output'].get('filename', 'model')+'_after_epoch_{}.ckpt'.format(epoch))

        current_time = time.time()
        torch.save({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_confusion': test_confusion,
            'classes': classes,
        }, config['output'].get('filename', 'model')+'_validation_after_epoch_{}.dat'.format(epoch))
        print('Epoch [{}/{}] completed, time since start {}, time this epoch {}, total remaining {}'
            .format(epoch + 1, num_epochs, pretty_print_time(current_time-start_time), pretty_print_time(current_time-start_epoch), 
                pretty_time_left(start_time, epoch+1, num_epochs)))

    # Test the model
    #losses, top1, confusion = validate( model, criterion, num_classes, test_loader )

      # Save the model checkpoint
    torch.save({
        'epoch': num_epochs + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, config['output'].get('filename', 'model')+'.ckpt')
    torch.save({
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_confusion': test_confusion,
        'classes': classes,
    }, config['output'].get('filename', 'model')+'_validation.dat')    


if __name__ == '__main__':
    main()


