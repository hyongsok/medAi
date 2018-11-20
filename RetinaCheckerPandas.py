import configparser
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import PandasDataset
from helper_functions import AccuracyMeter, AverageMeter
from time_left import pretty_print_time, pretty_time_left


def single_output_performance( labels, outputs, feature_number=5 ):
    predicted = nn.Sigmoid()(outputs)
    perf2 = (predicted[:,feature_number].round()==labels[:,feature_number])
    num_correct = float(perf2.sum())
    return num_correct

class RetinaCheckerPandas():
    """Deep learning model
    """

    def __init__( self ):
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

        self.train_file = None
        self.train_root = None
        self.test_file = None
        self.test_root = None

        self.train_dataset = None
        self.test_dataset = None
        self.num_classes = None
        self.classes = None
        self.normalize_data = True

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

    def initialize( self, config ):
        if config is None:
            raise ValueError('config cannot be None')
        elif config.__class__ == str:
            self.config = configparser.ConfigParser()
            self.config.read(config)
        else:
            self.config = config

        self.model_name = self.config['network'].get('model', 'resnet18')
        self.optimizer_name = self.config['network'].get('optimizer', 'Adam')
        self.criterion_name = self.config['network'].get('criterion', 'CrossEntropy')
        self.train_root = self.config['files'].get('train root', './train')
        self.train_file = self.config['files'].get('train file', 'labels.csv')
        self.test_root = self.config['files'].get('test root', '')
        self.test_file = self.config['files'].get('test file', '.')

        self.model_pretrained = self.config['network'].getboolean('pretrained', False)
        if self.device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device', self.device)
        self.initialized = True   


    def initialize_optimizer( self, **kwargs ):
        optimizer_loader = None
        if self.optimizer_name in torch.optim.__dict__.keys():
            optimizer_loader = torch.optim.__dict__[self.optimizer_name]
        else:
            warnings.warn('Could not identify optimizer')
            return

        self.learning_rate = self.config['hyperparameter'].getfloat('learning rate', 0.01)
        self.optimizer = optimizer_loader(self.model.parameters(), lr=self.learning_rate, **kwargs)

    def initialize_criterion( self, **kwargs ):
        criterion_loader = None
        if self.criterion_name in nn.__dict__.keys():
            criterion_loader = nn.__dict__[self.criterion_name]
        else:
            warnings.warn('Could not identify criterion')
            return

        self.criterion = criterion_loader(**kwargs)

    def initialize_model( self, **kwargs ):
        model_loader = None
        if self.model_name in models.__dict__.keys():
            model_loader = models.__dict__[self.model_name]
        else:
            warnings.warn('Could not identify model')
            return
        
        self.model = model_loader( pretrained=self.model_pretrained, **kwargs)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model = self.model.to(self.device)

    def load_datasets( self, normalize_factors=None, test_size=0.1 ):
        '''Loads the data sets from the path given in the config file
        '''

        if self.config['input'].getboolean('evaluation only', False):
            test_transform = self._get_test_transform(normalize_factors)
            self.test_dataset = PandasDataset.PandasDataset(source=self.test_file, mode='csv',
                                            root=self.test_root,
                                            transform=test_transform, target_transform=None)
        else:
            test_transform = self._get_test_transform(normalize_factors)
            train_transform = self._get_training_transform(normalize_factors)
            
            if not os.path.isfile(self.test_file): 
                dataset = PandasDataset.PandasDataset(source=self.train_file, mode='csv', root=self.train_root)
                self.train_dataset, self.test_dataset = dataset.split(test_size=test_size)
                self.train_dataset.transform = train_transform
                self.test_dataset.transform = test_transform
            else:
                self.train_dataset = PandasDataset.PandasDataset(source=self.train_file, mode='csv',
                                                            root=self.train_root,
                                                            transform=train_transform, target_transform=None)
                self.test_dataset = PandasDataset.PandasDataset(source=self.test_file, mode='csv',
                                                            root=self.test_root,
                                                            transform=test_transform, target_transform=None)

        self.classes = self.test_dataset.class_to_idx
        self.num_classes = len(self.classes)

    
    def create_dataloader( self, num_workers=8 ):
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
                                                    sampler=train_sampler,
                                                    num_workers=num_workers)

        test_sampler = None

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                sampler=test_sampler,
                                                num_workers=num_workers)

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
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(self.config['input'].get('checkpoint'), checkpoint['epoch']))
        except OSError as e:
            print("Exception occurred. Did not load model, starting from scratch.\n", e)


    def _get_training_transform( self, normalize_factors=None ):
        image_size = self.config['files'].getint('image size', 299)

        rotation_angle = self.config['transform'].getint('rotation angle', 180)
        rotation = transforms.RandomRotation(rotation_angle)
        
        brightness = self.config['transform'].getint('brightness', 0)
        contrast = self.config['transform'].getint('contrast', 0)
        saturation = self.config['transform'].getint('saturation', 0)
        hue = self.config['transform'].getint('hue', 0)
        color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        
        transform_list = [
                color_jitter,
                rotation,
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=image_size, scale=(0.25,1.0), ratio=(1,1)),
                transforms.ToTensor(),
            ]
        
        if not normalize_factors is None:
            normalize = transforms.Normalize(mean=normalize_factors[0],
                                            std=normalize_factors[1])
            transform_list.append(normalize)

        train_transform = transforms.Compose(transform_list)

        return train_transform
    
    def _get_test_transform( self, normalize_factors=None ):
        image_size = self.config['files'].getint('image size', 299)
        # normalization factors for the DMR dataset were manually derived

        transform_list = [
                transforms.Resize(size=int(image_size*1.1)),
                transforms.CenterCrop(size=image_size),
                transforms.ToTensor(),
            ]
        
        if not normalize_factors is None:
            normalize = transforms.Normalize(mean=normalize_factors[0],
                                            std=normalize_factors[1])
            transform_list.append(normalize)

        test_transform = transforms.Compose(transform_list)

        return test_transform

    def train( self, evaluate_performance=single_output_performance ):
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
        accuracy = AccuracyMeter()
        self.model.train()

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = self.criterion(outputs, labels)

            # store results & evaluate accuracy
            losses.update(loss.item(), images.size(0))

            num_correct = evaluate_performance( labels, outputs )
            accuracy.update(num_correct, labels.size(0))

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            current_time = time.time()
            print('Epoch [{}], Step [{}/{}], Loss: {:.4f},  Samples: {}, Correct: {} ({:.1f}%),  time in epoch {}, epoch remaining {}'
                .format(self.epoch + 1, i + 1, len(self.train_loader), loss.item(), labels.size(0), num_correct, 
                    num_correct/labels.size(0)*100,
                    pretty_print_time(current_time-start_time_epoch), 
                    pretty_time_left(start_time_epoch, i+1, len(self.train_loader))))
        print('Epoch learning completed. Training accuracy {:.1f}%'.format(accuracy.avg*100))

        self.epoch += 1
        return losses, accuracy   


    def validate(self, test_loader = None, evaluate_performance=single_output_performance ):
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
            losses = AverageMeter()
            accuracy = AccuracyMeter()

            #confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float)
            confusion = torch.zeros((2, 2), dtype=torch.float)

            for images, labels in test_loader:
                print(np.unique(labels, return_counts=True))
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), images.size(0))

                num_correct = evaluate_performance( labels, outputs )

                accuracy.update(num_correct, labels.size(0))
                predicted = torch.nn.Sigmoid()(outputs).round()

                for pred, lab in zip(predicted[:,5], labels[:,5]):
                    confusion[int(pred.item()), int(lab.item())] += 1

                print('Test - samples: {}, correct: {} ({:.1f}%), loss: {}'.format(labels.size(0), num_correct, num_correct/labels.size(0)*100, loss.item()))
                
        
        return losses, accuracy, confusion


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
        class_distribution = np.vstack([s for s in dataset.targets]).astype(np.int)
        weights = np.zeros(self.num_classes)
        for ii in range(self.num_classes):
            weights[ii] = np.sum(class_distribution[:,ii] == 1)/len(class_distribution)

        inverted_weights = (1/weights)/np.sum(1/weights)
        sampling_weights = np.zeros(class_distribution.shape[0], dtype=np.float)
        for ii in range(self.num_classes):
            sampling_weights[class_distribution[:,ii] == 1] = inverted_weights[ii]
        sampling_weights /= sampling_weights.sum()


        sampler = torch.utils.data.WeightedRandomSampler( sampling_weights, num_samples, True )
        return sampler
