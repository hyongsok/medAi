import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import configparser
import sys, os, glob
import time
from time_left import pretty_time_left, pretty_print_time
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value
    
    Usage: 
    am = AverageMeter()
    am.update(123)
    am.update(456)
    am.update(789)
    
    last_value = am.val
    average_value = am.avg
    
    am.reset() #set all to 0"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
    """Computes and stores the correctly classified samples
    
    Usage: pass the number of correct (val) and the total number (n)
    of samples to update(val, n). Then .avg contains the 
    percentage correct.
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class RetinaChecker(object):
    """[summary]
    
    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self):
        self.device = None
        self.config = None

        self.model = None
        self.optimizer = None
        self.criterion = None

        self.train_dataset = None
        self.test_dataset = None
        self.num_classes = None
        self.classes = None

        self.train_loader = None
        self.test_loader = None

        self.start_epoch = None
        self.epoch = None

        self.initialized = False
    
    def train( self ):
        '''Deep learning training function to optimize the network with all images in the train_loader.
        
        Arguments:
            model {torch.nn.Module} -- the deep neural network
            criterion {torch.nn._Loss} -- the loss function, e.g. cross entropy or negative log likelihood
            optimizer {torch.optim.Optimizer} -- the optimizer to use for the training, e.g. Adam or SGD
            train_loader {torch.utils.data.DataLoader} -- contains the data for training
            device {torch.device} -- target computation device        
            epoch {int} -- the number of the current epoch (for console output only)
        
        Returns:
            AverageMeter -- training loss
            AccuracyMeter -- training accuracy
        '''

        start_time_epoch = time.time()
        losses = AverageMeter()
        top1 = AccuracyMeter()
        self.model.train()

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # evaluate
            losses.update(loss.item(), images.size(0))
            _, predicted = torch.max(outputs.data, 1)
            top1.update((predicted == labels).sum().item(), labels.size(0))

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_time = time.time()
            print('Epoch [{}], Step [{}/{}], Loss: {:.4f},  Samples: {}, Correct: {} ({:.1f}%),  time in epoch {}, epoch remaining {}'
                .format(self.epoch + 1, i + 1, len(self.train_loader), loss.item(), labels.size(0), (predicted == labels).sum().item(), 
                    (predicted == labels).sum().item()/labels.size(0)*100,
                    pretty_print_time(current_time-start_time_epoch), 
                    pretty_time_left(start_time_epoch, i+1, len(self.train_loader))))
        print('Epoch learning completed. Training accuracy {:.1f}%'.format(top1.avg*100))

        self.epoch += 1
        return losses, top1   



    def validate( self, test_loader = None ):
        '''Evaluates the model given the criterion and the data in test_loader
        
        Arguments:
            model {torch.nn.Module} -- the deep neural network
            criterion {torch.nn._Loss} -- the loss function, e.g. cross entropy or negative log likelihood
            num_classes {int} -- the number of test classes
            train_loader {torch.utils.data.DataLoader} -- contains the data for training
            device {torch.device} -- target computation device        
        
        Returns:
            AverageMeter -- training loss
            AccuracyMeter -- training accuracy
            numpy.Array -- [num_classes, num_classes] confusion matrix, columns are true classes, rows predictions
        '''

        if test_loader is None:
            test_loader = self.test_loader

        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            losses = AverageMeter()
            top1 = AccuracyMeter()

            confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float)

            for images, labels in test_loader:
                print(np.unique(labels, return_counts=True))
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), images.size(0))

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                top1.update((predicted == labels).sum().item(), labels.size(0))
                print('Test - samples: {}, correct: {} ({:.1f}%), loss: {}'.format(labels.size(0), (predicted == labels).sum().item(), top1.avg*100, loss.item()))
                for pred, lab in zip(predicted, labels):
                    confusion[pred, lab] += 1
        
        return losses, top1, confusion

    def _get_optimizer( self, name ):
        optimizer = None
        if name == 'Adam':
            optimizer = torch.optim.Adam
        else:
            print('Could not identify optimizer')
        return optimizer

    def _get_criterion( self, name ):
        criterion = None
        if name == 'CrossEntropy':
            criterion = nn.CrossEntropyLoss
        else:
            print('Could not identify criterion')
        return criterion

    def _get_model( self, name ):
        model = None
        if name.startswith('resnet'):
            if name.endswith('18'):
                model = models.resnet18
            elif name.endswith('34'):
                model = models.resnet34
            elif name.endswith('50'):
                model = models.resnet50
            else:
                model = models.resnet18
        else:
            print('Could not identify model')
        return model

    def initialize_deep_learning_model( self ):
        """Generates the model, criterion, and optimizer
        
        Arguments:
            config {configparser.ConfigParser} -- configuration file
            num_classes {int} -- number of target classes (= output dimensions) of the model
            device {torch.device} -- target device (cpu/gpu/...)
        
        Returns:
            torch.optim.Optimizer -- the optimizer to use for the training, e.g. Adam or SGD
            torch.nn._Loss -- the loss function, e.g. cross entropy or negative log likelihood
            torch.nn.Module -- the deep neural network  
        """

        learning_rate = self.config['hyperparameter'].getfloat('learning rate', 0.01)
        
        # Deep learning model resnet18 without prior training on ImageNet data.
        # be aware that if you change the model there might be additional parameter
        # necessary, e.g. for inception you need aux_logits as parameter
        self.model = self._get_model('resnet18')(pretrained=False, num_classes=self.num_classes)

        # Reducing the output layer to num_classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = self._get_criterion('CrossEntropy')()
        self.optimizer = self._get_optimizer('Adam')(self.model.parameters(), lr=learning_rate)

    def _load_datasets( self ):
        '''Loads the data sets from the path given in the config file
        
        Arguments:
            config {configparser.ConfigParser} -- the configuration read from ini file
        
        Returns:
            torch.utils.data.Dataset -- training data
            torch.utils.data.Dataset -- Test data
        '''

        image_size = self.config['files'].getint('image size', 299)
        # normalization factors for the DMR dataset were manually derived
        normalize = transforms.Normalize(mean=[0.3198, 0.1746, 0.0901],
                                        std=[0.2287, 0.1286, 0.0723])

        rotation_angle = self.config['transform'].getint('rotation angle', 180)
        rotation = transforms.RandomRotation(rotation_angle)
        
        brightness = self.config['transform'].getint('brightness', 0)
        contrast = self.config['transform'].getint('contrast', 0)
        saturation = self.config['transform'].getint('saturation', 0)
        hue = self.config['transform'].getint('hue', 0)
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

        self.train_dataset = torchvision.datasets.ImageFolder(root=self.config['files'].get('train path', './train'),
                                                        transform=train_transform)
        self.test_dataset = torchvision.datasets.ImageFolder(root=self.config['files'].get('test path', './test'),
                                                        transform=test_transform)
        
        self.classes = self.train_dataset.class_to_idx
        self.num_classes = len(self.classes)

    def _create_dataloader( self ):
        """Generates the dataloader (and their respective samplers) for
        training and test data from the training and test data sets.
        Sampler for training data is an unbiased sampler for all classes
        in the training set, i.e. even if the class distribution in the
        data set is biased, all classes are equally contained in the sampling.
        No specific sampler for test data.
        
        Arguments:
            train_dataset {torch.util.data.Dataset} -- training data
            test_dataset {torch.util.data.Dateset} -- test data
            config {configparser.ConfigParser} -- configuration file containing 
            batch size and number of samples
        
        Returns:
            torch.util.data.DataLoader -- loader for training data
            torch.util.data.DataLoader -- loader for test data
        """

        batch_size = self.config['hyperparameter'].getint('batch size', 10)
        num_samples = self.config['files'].getint('samples', 1000)
        
        train_sampler = get_sampler( self.train_dataset, num_samples )
        test_sampler = None

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                sampler=train_sampler)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                sampler=test_sampler)


    def initialize( self, config ):
        self.config = config

        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        # Loading data sets based on configuration
        self._load_datasets()
        self.start_epoch = 0
        self.epoch = 0

        # Initializing sampler and data (=patch) loader
        self._create_dataloader()

        # Initialize the model
        self.initialize_deep_learning_model()

    def save_state( self, train_loss, train_accuracy, test_loss, test_accuracy, test_confusion, filename ):
        """Save the current state of the model including history of training and test performance
        
        Arguments:
            model {torch.nn.Module} -- the deep neural network
            optimizer {torch.optim.Optimizer} -- the optimizer to use for the training, e.g. Adam or SGD
            num_epochs {int} -- the number of executed epochs  
            train_loss {torch.Array} -- tensor of training losses
            train_accuracy {torch.Array} -- tensor of training accuracy
            test_loss {torch.Array} -- tensor of test losses
            test_accuracy {torch.Array} -- tensor of test accuracy
            test_confusion {torch.Array} -- tensor of confusion matrices
            classes {dict} -- class dictionary
            filename {string} -- target filename
        """

        torch.save({
            'epoch': self.epoch+1,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_confusion': test_confusion,
            'classes': self.classes,
        }, filename)

    def load_state( self ):
        """Load the state stored in the config into the given model and optimizer.
        Model and optimizer must match exactly to the stored model, will crash
        otherwise.
        
        Arguments:
            model {torch.nn.Module} -- the deep neural network
            optimizer {torch.optim.Optimizer} -- the optimizer to use for the training, e.g. Adam or SGD
            config {configparser.ConfigParser} -- configuration file
        
        Returns:
            torch.nn.Module -- the deep neural network      
            torch.optim.Optimizer -- the optimizer to use for the training, e.g. Adam or SGD
            int -- number of epochs trained
        """
        try:
            checkpoint = torch.load(self.config['input'].get('checkpoint'))
            self.start_epoch = checkpoint['epoch']
            self.epoch = self.start_epoch
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(self.config['input'].get('checkpoint'), checkpoint['epoch']))
        except OSError as e:
            print("Exception occurred. Did not load model, starting from scratch.\n", e)


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

def get_sampler( dataset, num_samples ):
    '''The distribution of samples in training and test data is not equal, i.e.
the normal class is over represented. To get an unbiased sample (for example with 5
classes each class should make up for about 20% of the samples) we calculate the 
probability of each class in the source data (=weights) then we invert the weights 
and assign to each sample in the class the inverted probability (normalized to 1 over
all samples) and use this as the probability for the weighted random sampler.
    
    Arguments:
        dataset {torch.util.data.Dataset} -- the dataset to sample from
        num_samples {int} -- number of samples to be drawn bei the sampler

    Returns:
        torch.util.data.Sampler -- Sampler object from which patches for the training
        or evaluation can be drawn
    '''
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


    sampler = torch.utils.data.WeightedRandomSampler( sampling_weights, num_samples, True )
    return sampler



def print_dataset_stats( dataset, loader ):
    """Debug / support function. Iterates through all samples in the dataloader
    and prints the distribution of classes.
    This really reads all data from the data set and creates all samples. Depending
    on your hardware and sample size this can take time. A lot of time. Use only for
    debugging or analysis.
    
    Arguments:
        dataset {torch.util.data.DataSet} -- Source data set, required for class labels
        loader {torch.util.data.DataLoader} -- Data loader to evaluate
    """

    classes = dataset.class_to_idx
    labels = np.zeros(len(classes))
    for _, labels in loader:
        lab, count = np.unique(labels, return_counts=True)
        labels[lab] += count
    print('Samples:', labels.sum())
    for key, val in classes.items():
        print('{}: {} - {} samples ({:.1f}%)'.format(val, key, labels[int(val)], labels[int(val)]/labels.sum()*100))
        

def initialize_meters( config, start_epoch, num_epochs, num_classes ):
    """Initialize the accuracy and loss meters and confusion matrices
    
    Arguments:
        config {configparder.ConfigParser} -- configuration file
        start_epoch {int} -- previously executed epochs
        num_epochs {int} -- number of new epochs
        num_classes {int} -- number of classes (= output dimensions)
    
    Returns:
        torch.Array -- training loss tensor
        torch.Array -- training accuracy tensor
        torch.Array -- test loss tensor
        torch.Array -- test accuracy tensor
        torch.array -- tensor of shape (start_epoch+num_epochs, num_classes, num_classes) for confusion matrix
    """

    train_loss = torch.zeros(num_epochs, dtype=torch.float)
    train_accuracy = torch.zeros(num_epochs, dtype=torch.float)
    test_loss = torch.zeros(num_epochs, dtype=torch.float)
    test_accuracy = torch.zeros(num_epochs, dtype=torch.float)
    test_confusion = torch.zeros((num_epochs, num_classes, num_classes), dtype=torch.float)

    if config['input'].getboolean('resume', False):
        checkpoint = torch.load(config['input'].get('checkpoint'))
        train_loss[:start_epoch] = checkpoint['train_loss']
        train_accuracy[:start_epoch] = checkpoint['train_accuracy']
        test_loss[:start_epoch] = checkpoint['test_loss']
        test_accuracy[:start_epoch] = checkpoint['test_accuracy']
        test_confusion[:start_epoch,:,:] = checkpoint['test_confusion']
    return train_loss, train_accuracy, test_loss, test_accuracy, test_confusion


def reduce_to_2_classes( confusion, class_groups ):
    """Reduces the confusion matrix from nxn to 2x2. All classes in the 2
    class groups will be treated as one class in the new confusion matrix. For 
    example if a 5x5 confusion matrix with classes 0..4 should be reduced to a
    2x2 confusion matrix where classes 0,1 are the first column and classes 2,3,4
    are the second column, the class_groups argument would be [[0,1],[2,3,4]].
    Confusion matrix must have the true classes in columns and the predictions in rows.
    
    Arguments:
        confusion {numpy.Array} -- confusion matrix with true classes in columns
        class_groups {list} -- list with 2 lists containing the classes to be merged
    
    Returns:
        numpy.Array -- 2x2 confusion matrix with true classes in columns
    """
    # first build the sum the columns of the 2 groups to reduce to 2 (n,) arrays
    # then stack them up transposed as (2,n) and repeat the sum along the columns
    # to 2 (2,) arrays. Stack them up transposed to get the correct confusion matrix
    confusion_2class = np.vstack((np.vstack((confusion[:,class_groups[0]].sum(1),
                                         confusion[:,class_groups[1]].sum(1)))[:,class_groups[0]].sum(1), 
                                np.vstack((confusion[:,class_groups[0]].sum(1),
                                         confusion[:,class_groups[1]].sum(1)))[:,class_groups[1]].sum(1))
                                )
    return confusion_2class



def save_performance( train_loss, train_accuracy, test_loss, test_accuracy, test_confusion, classes, filename ):
    """Save the current model performance, e.g. for visualization
    
    Arguments:
        train_loss {torch.Array} -- tensor of training losses
        train_accuracy {torch.Array} -- tensor of training accuracy
        test_loss {torch.Array} -- tensor of test losses
        test_accuracy {torch.Array} -- tensor of test accuracy
        test_confusion {torch.Array} -- tensor of confusion matrices
        classes {dict} -- class dictionary
        filename {string} -- target filename
    """
    torch.save({
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_confusion': test_confusion,
        'classes': classes,
        }, filename)    

def main():

    # Reading configuration file
    config = configparser.ConfigParser()
    if len(sys.argv) > 1:
        config.read(sys.argv[1])
    else:
        config.read('default.cfg')

    rc = RetinaChecker()
    rc.initialize( config )

    # Patch statistics requires the data loader to load all patches once
    # to analyze the data. This takes a lot of time. Use only when you
    # change something on the sampler to see the results. Should be False
    # for regular learning passes.
    if config['files'].getboolean('show patch stats', False):
        print_dataset_stats( rc.train_dataset, rc.train_loader )
        print_dataset_stats( rc.test_dataset, rc.test_loader )
    print('Data loaded. Setting up model.')
    
    # loading previous models
    if config['input'].getboolean('resume', False):
        rc.load_state()

    print('Model set up. Ready to train.')

    # Performance meters initalized (either empty or from file)
    num_epochs = rc.start_epoch + config['hyperparameter'].getint('epochs', 10)
    train_loss, train_accuracy, test_loss, test_accuracy, test_confusion = initialize_meters( config, rc.start_epoch, num_epochs, rc.num_classes )

    # Starting training & evaluation
    start_time = time.time()
    best_accuracy = 0
    for epoch in range(rc.start_epoch, num_epochs):
        start_time_epoch = time.time()

        # Train the model and record training loss & accuracy
        losses, top1 = rc.train( )
        train_loss[epoch] = losses.avg
        train_accuracy[epoch] = top1.avg
        
        losses, top1, confusion = rc.validate()
        test_loss[epoch] = losses.avg
        test_accuracy[epoch] = top1.avg
        test_confusion[epoch,:,:] = confusion

        print('Test Accuracy of the model on the {} test images: {} %'.format(top1.count, top1.avg*100))
        print('Classes: {}'.format(rc.classes))
        print('Confusion matrix:\n', (confusion))

        confusion_2class = reduce_to_2_classes( confusion, [(1,3), (0,2,4)])
        print('Accuracy: {:.1f}%'.format(np.diag(confusion_2class).sum()/confusion_2class.sum()*100))
        print(confusion_2class)
        print('Sensitivity: {:.1f}%'.format(confusion_2class[1,1]/confusion_2class[:,1].sum()*100))
        print('Specificity: {:.1f}%'.format(confusion_2class[0,0]/confusion_2class[:,0].sum()*100))

        if config['output'].getboolean('save during training', False) and ((epoch+1) % config['output'].getint('save every nth epoch', 10) == 0):
            rc.save_state( train_loss[:(epoch+1)], 
                        train_accuracy[:(epoch+1)], test_loss[:(epoch+1)], 
                        test_accuracy[:(epoch+1)], test_confusion[:(epoch+1),:,:], 
                        config['output'].get('filename', 'model')+'_after_epoch_{}'.format(epoch+1)+config['output'].get('extension', '.ckpt') )
        
        if top1.avg > best_accuracy:
            rc.save_state( train_loss[:(epoch+1)], 
                        train_accuracy[:(epoch+1)], test_loss[:(epoch+1)], 
                        test_accuracy[:(epoch+1)], test_confusion[:(epoch+1),:,:], 
                        config['output'].get('filename', 'model')+'_best_accuracy'+config['output'].get('extension', '.ckpt') )
            best_accuracy = top1.avg

        save_performance( train_loss[:(epoch+1)], train_accuracy[:(epoch+1)], test_loss[:(epoch+1)], 
                test_accuracy[:(epoch+1)], test_confusion[:(epoch+1),:,:], rc.classes, 
                config['output'].get('filename', 'model')+'_validation_after_epoch_{}.dat'.format(epoch+1) )

        # Output on progress
        current_time = time.time()
        print('Epoch [{}/{}] completed, time since start {}, time this epoch {}, total remaining {}, validation in {}'
            .format(epoch + 1, num_epochs, pretty_print_time(current_time-start_time), pretty_print_time(current_time-start_time_epoch), 
                pretty_time_left(start_time, epoch+1-rc.start_epoch, num_epochs-rc.start_epoch), 
                config['output'].get('filename', 'model')+'_validation_after_epoch_{}.dat'.format(epoch+1)))


    # Save the model checkpoint
    rc.save_state( train_loss, train_accuracy, test_loss, test_accuracy, test_confusion, 
                config['output'].get('filename', 'model')+config['output'].get('extension', '.ckpt') )

    # cleanup
    print('Cleaning up...')
    delete_files = glob.glob(config['output'].get('filename', 'model')+'_validation_after_epoch_*.dat')
    if config['output'].getboolean('save during training', False):
        delete_files += glob.glob(config['output'].get('filename', 'model')+'_after_epoch_*'+config['output'].get('extension', '.ckpt'))
    
    for f in delete_files:
        try:
            print('deleting', os.path.join(os.path.abspath(os.curdir), f))
            os.remove(os.path.join(os.path.abspath(os.curdir), f))
        except OSError as e:
            print(e)

if __name__ == '__main__':
    main()


