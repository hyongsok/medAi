import configparser
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import time
from time_left import pretty_time_left, pretty_print_time
import numpy as np

from helper_functions import AverageMeter, AccuracyMeter

class RetinaChecker(object):
    """Deep learning model
    """

    def __init__(self):
        self.device = None
        self.config = None

        self.model = None
        self.model_name = 'resnet18'
        self.optimizer = None
        self.optimizer_name = 'Adam'
        self.criterion = None
        self.criterion_name = 'CrossEntropy'

        self.model_pretrained = False
        self.model_kwargs = {}

        self.train_dataset = None
        self.test_dataset = None
        self.num_classes = None
        self.classes = None

        self.train_loader = None
        self.test_loader = None

        self.start_epoch = 0
        self.epoch = 0

        self.learning_rate = None

        self.initialized = False
    
    def __str__( self ):
        desc = 'RetinaChecker\n'
        if self.initialized:
            desc += 'Network: ' + self.model_name
            if self.model_pretrained:
                desc += ' (pretrained)\n'
            else:
                desc += '\n'
            desc += 'Optimizer: ' + self.optimizer_name + '\n'
            desc += 'Criterion: ' + self.criterion_name + '\n'
            desc += 'Epochs: ' + self.epoch + '\n'
        else:
            desc += 'not initialized'
        return desc
    
    def train( self ):
        '''Deep learning training function to optimize the network with all images in the train_loader.
        
        Returns:
            AverageMeter -- training loss
            AccuracyMeter -- training accuracy
        '''
        if not self.initialized:
            print('RetinaChecker not initialized.')
            return

        if self.train_loader is None:
            print('No training loader defined. Check configuration.')
            return

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
            test_loader {torch.utils.data.DataLoader} -- contains the data for training, if None takes internal test_loader     
        
        Returns:
            AverageMeter -- training loss
            AccuracyMeter -- training accuracy
            numpy.Array -- [num_classes, num_classes] confusion matrix, columns are true classes, rows predictions
        '''
        if not self.initialized:
            print('RetinaChecker not initialized.')
            return

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

    def _initialize_optimizer( self ):
        optimizer_loader = None
        if self.optimizer_name in torch.optim.__dict__.keys():
            optimizer_loader = torch.optim.__dict__[self.optimizer_name]
        else:
            print('Could not identify optimizer')
            return

        self.learning_rate = self.config['hyperparameter'].getfloat('learning rate', 0.01)
        self.optimizer = optimizer_loader(self.model.parameters(), lr=self.learning_rate)

    def _initialize_criterion( self ):
        criterion_loader = None
        if self.criterion_name in nn.__dict__.keys():
            criterion_loader = nn.__dict__[self.criterion_name]
        else:
            print('Could not identify criterion')
            return

        self.criterion = criterion_loader()

    def _initialize_model( self ):
        model_loader = None
        if self.model_name in models.__dict__.keys():
            model_loader = models.__dict__[self.model_name]
        else:
            print('Could not identify model')
            return
        
        self.model = model_loader( pretrained=self.model_pretrained, num_classes=self.num_classes, **self.model_kwargs)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model = self.model.to(self.device)

    def _load_datasets( self ):
        '''Loads the data sets from the path given in the config file
        '''

        if not self.config['input'].getboolean('evaluation only', False):
            self._load_training_data()

        self._load_test_data()

    def _load_training_data( self ):
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

        self.train_dataset = torchvision.datasets.ImageFolder(root=self.config['files'].get('train path', './train'),
                                                        transform=train_transform)
        
        self.classes = self.train_dataset.class_to_idx
        self.num_classes = len(self.classes)
    
    def _load_test_data( self ):
        image_size = self.config['files'].getint('image size', 299)
        # normalization factors for the DMR dataset were manually derived
        normalize = transforms.Normalize(mean=[0.3198, 0.1746, 0.0901],
                                        std=[0.2287, 0.1286, 0.0723])

        test_transform = transforms.Compose([
                transforms.Resize(size=int(image_size*1.1)),
                transforms.CenterCrop(size=image_size),
                transforms.ToTensor(),
                normalize,
            ])

        self.test_dataset = torchvision.datasets.ImageFolder(root=self.config['files'].get('test path', './test'),
                                                        transform=test_transform)
        
        self.classes = self.test_dataset.class_to_idx
        self.num_classes = len(self.classes)

    def _create_dataloader( self ):
        """Generates the dataloader (and their respective samplers) for
        training and test data from the training and test data sets.
        Sampler for training data is an unbiased sampler for all classes
        in the training set, i.e. even if the class distribution in the
        data set is biased, all classes are equally contained in the sampling.
        No specific sampler for test data.
        """

        batch_size = self.config['hyperparameter'].getint('batch size', 10)
        
        if not self.config['input'].getboolean('evaluation only', False):
            num_samples = self.config['files'].getint('samples', 1000)
        
            train_sampler = self._get_sampler( self.train_dataset, num_samples )

            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=train_sampler)

        test_sampler = None

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                sampler=test_sampler)

    def _process_config( self ):
        """Parses the config file stored in the config member and sets the
        member variables accordingly. Important before initializing model,
        optimizer, or criterion.
        """

        self.model_name = self.config['network'].get('model', 'resnet18')
        self.optimizer_name = self.config['network'].get('optimizer', 'Adam')
        self.criterion_name = self.config['network'].get('criterion', 'CrossEntropy')

        self.model_pretrained = self.config['network'].getboolean('pretrained', False)

    def initialize( self, config ):
        """Initializes the RetinaChecker from given configuration file. Sets the
        device (first gpu if cuda is available, cpu otherwise), loads the data sets,
        creates the data loaders and initializes model, criterion, and optimizer.
        After this the RetinaChecker is ready for loading a previous state, training,
        and evaluation.
        
        Arguments:
            config {configparser.ConfigParser} -- configuration file 
        """
        if config is None:
            raise ValueError('config cannot be None')
        elif config.__class__ == str:
            self.config = configparser.ConfigParser()
            self.config.read(config)
        else:
            self.config = config

        self._process_config()

        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device', self.device)

        # Loading data sets based on configuration
        self._load_datasets()

        # Initializing sampler and data (=patch) loader
        self._create_dataloader()

        # Initialize the model
        self._initialize_model()
        self._initialize_criterion()
        self._initialize_optimizer()

        self.initialized = True
        

    def save_state( self, train_loss, train_accuracy, test_loss, test_accuracy, test_confusion, filename ):
        """Save the current state of the model including history of training and test performance
        
        Arguments:
            train_loss {torch.Array} -- tensor of training losses
            train_accuracy {torch.Array} -- tensor of training accuracy
            test_loss {torch.Array} -- tensor of test losses
            test_accuracy {torch.Array} -- tensor of test accuracy
            test_confusion {torch.Array} -- tensor of confusion matrices
            filename {string} -- target filename
        """

        torch.save({
            'epoch': self.epoch,
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
        """
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(self.config['input'].get('checkpoint'))
            else:
                checkpoint = torch.load(self.config['input'].get('checkpoint'), map_location='cpu')
            self.start_epoch = checkpoint['epoch']
            self.epoch = self.start_epoch
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(self.config['input'].get('checkpoint'), checkpoint['epoch']))
        except OSError as e:
            print("Exception occurred. Did not load model, starting from scratch.\n", e)


    def _get_sampler( self, dataset, num_samples ):
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
        class_distribution = np.array(dataset.imgs)[:,1].astype(int)
        weights = np.zeros(self.num_classes)
        for ii in range(self.num_classes):
            weights[ii] = np.sum(class_distribution == ii)/len(class_distribution)

        inverted_weights = (1/weights)/np.sum(1/weights)
        sampling_weights = np.zeros(class_distribution.shape, dtype=np.float)
        for ii in range(self.num_classes):
            sampling_weights[class_distribution == ii] = inverted_weights[ii]
        sampling_weights /= sampling_weights.sum()


        sampler = torch.utils.data.WeightedRandomSampler( sampling_weights, num_samples, True )
        return sampler





