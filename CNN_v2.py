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

class AccuracyMeter(object):
    """Computes and stores the correctly classified samples"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


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
    top1 = AccuracyMeter()
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # evaluate
        losses.update(loss.item(), images.size(0))
        _, predicted = torch.max(outputs.data, 1)
        top1.update((predicted == labels).sum().item(), labels.size(0))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_time = time.time()
        print('Epoch [{}], Step [{}/{}], Loss: {:.4f},  Samples: {}, Correct: {} ({:.1f}%), time since start {}, time in epoch {}, epoch remaining {}'
            .format(epoch + 1, i + 1, len(train_loader), loss.item(), labels.size(0), (predicted == labels).sum().item(), 
                (predicted == labels).sum().item()/labels.size(0)*100,
                pretty_print_time(current_time-start_time), pretty_print_time(current_time-start_epoch), 
                pretty_time_left(start_epoch, i+1, len(train_loader))))
    print('Epoch learning completed. Training accuracy {:.1f}%'.format(top1.avg*100))

    return losses, top1   

def validate( model, criterion, num_classes, test_loader, device ):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        losses = AverageMeter()
        top1 = AccuracyMeter()

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
            print('Test - samples: {}, correct: {} ({:.1f}%), loss: {}'.format(labels.size(0), (predicted == labels).sum().item(), top1.avg*100, loss.item()))
            for pred, lab in zip(predicted, labels):
                confusion[pred, lab] += 1
    
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

    print("Paremeters:\nEpochs:\t\t{num_epochs}\nBatch size:\t{batch_size}\nLearning rate:\t{learning_rate}".format(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate))

    # normalization factors for the DMR dataset were manually derived
    normalize = transforms.Normalize(mean=[0.3198, 0.1746, 0.0901],
                                     std=[0.2287, 0.1286, 0.0723])

    rotation_angle = config['transform'].getint('rotation angle', 180)
    rotation = transforms.RandomRotation(rotation_angle)
    
    brightness = config['transform'].getint('brightness', 0)
    contrast = config['transform'].getint('contrast', 0)
    saturation = config['transform'].getint('saturation', 0)
    hue = config['transform'].getint('hue', 0)
    color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    train_transform = transforms.Compose([
            color_jitter,
            rotation,
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=image_size, scale=(0.25,1.0), ratio=(1,1)),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
            transforms.Resize(size=int(image_size*1.1)),
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.ImageFolder(root=config['files'].get('train path', './train'),
                                                    transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=config['files'].get('test path', './test'),
                                                    transform=test_transform)
    
    # The distribution of samples in training and test data is not equal, i.e.
    # the normal class is over represented. To get an unbiased sample (each of 
    # the 4 classes has probability 0.25) we calculate the probability of each
    # class in the source data (=weights) then we invert the weights and assign
    # to each sample in the class the inverted probability (normalized to 1 over
    # all samples) and use this as the probability for the weighted random sampler
    classes = train_dataset.class_to_idx
    class_distribution = np.array(train_dataset.imgs)[:,1].astype(int)
    weights = np.zeros(len(classes))
    for ii in range(len(classes)):
        weights[ii] = np.sum(class_distribution == ii)/len(class_distribution)

    inverted_weights = (1/weights)/np.sum(1/weights)
    sampling_weights = np.zeros(class_distribution.shape, dtype=np.float)
    for ii in range(len(classes)):
        sampling_weights[class_distribution == ii] = inverted_weights[ii]
    sampling_weights /= sampling_weights.sum()


    train_sampler = torch.utils.data.WeightedRandomSampler( sampling_weights, num_samples, True )
    test_sampler = None

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
    
    if config['files'].getboolean('show patch stats', False):
        train_labels = np.zeros(len(classes))
        test_labels = np.zeros(len(classes))
        for _, labels in train_loader:
            lab, count = np.unique(labels, return_counts=True)
            train_labels[lab] += count
        for _, labels in test_loader:
            lab, count = np.unique(labels, return_counts=True)
            test_labels[lab] += count
        print('Training data:')
        print('Source data weights:', weights)
        print('Training samples:', train_labels.sum())
        for key, val in train_dataset.class_to_idx.items():
            print('{}: {} - {} samples ({:.1f}%)'.format(val, key, train_labels[int(val)], train_labels[int(val)]/train_labels.sum()*100))
            
        print('Test data:')
        print('Test samples:', test_labels.sum())
        for key, val in train_dataset.class_to_idx.items():
            print('{}: {} - {} samples ({:.1f}%)'.format(val, key, test_labels[int(val)], test_labels[int(val)]/test_labels.sum()*100))

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
        #adjust_learning_rate(optimizer, epoch, learning_rate)

        losses, top1 = train( train_loader, device, model, criterion, optimizer, epoch, start_time )
        train_loss[epoch] = losses.avg
        train_accuracy[epoch] = top1.avg
        
        losses, top1, confusion = validate( model, criterion, num_classes, test_loader, device )
        test_loss[epoch] = losses.avg
        test_accuracy[epoch] = top1.avg
        test_confusion[epoch,:,:] = confusion

        print('Test Accuracy of the model on the {} test images: {} %'.format(top1.count, top1.avg*100))
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
            'train_loss': train_loss[:(epoch+1)],
            'train_accuracy': train_accuracy[:(epoch+1)],
            'test_loss': test_loss[:(epoch+1)],
            'test_accuracy': test_accuracy[:(epoch+1)],
            'test_confusion': test_confusion[:(epoch+1),:,:],
            'classes': classes,
        }, config['output'].get('filename', 'model')+'_validation_after_epoch_{}.dat'.format(epoch))
        print('Epoch [{}/{}] completed, time since start {}, time this epoch {}, total remaining {}, validation in {}'
            .format(epoch + 1, num_epochs, pretty_print_time(current_time-start_time), pretty_print_time(current_time-start_epoch), 
                pretty_time_left(start_time, epoch+1, num_epochs), 
                config['output'].get('filename', 'model')+'_validation_after_epoch_{}.dat'.format(epoch)))

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


