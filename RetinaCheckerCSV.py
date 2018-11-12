import configparser
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
from time_left import pretty_time_left, pretty_print_time
import numpy as np

import RetinaCheckerMultiClass
import EnhancedImageFolder
from helper_functions import AverageMeter, AccuracyMeter

class RetinaCheckerCSV(RetinaCheckerMultiClass.RetinaCheckerMultiClass):
    """Deep learning model
    """

    def __init__( self ):
        super().__init__()
        self.csv_file = None

    def initialize( self, config ):
        if config is None:
            raise ValueError('config cannot be None')
        elif config.__class__ == str:
            self.config = configparser.ConfigParser()
            self.config.read(config)
        else:
            self.config = config

        self.csv_file = config['files'].get('label file', 'labels.csv')

        super().initialize( config )
   
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
        accuracy = AccuracyMeter()
        self.model.train()

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # store results & evaluate accuracy
            losses.update(loss.item(), images.size(0))

            num_correct = self._evaluate_performance( labels, outputs )
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

                num_correct = self._evaluate_performance( labels, outputs )

                accuracy.update(num_correct, labels.size(0))
                predicted = torch.nn.Sigmoid()(outputs).round()

                for pred, lab in zip(predicted[:,5], labels[:,5]):
                    confusion[int(pred.item()), int(lab.item())] += 1

                print('Test - samples: {}, correct: {} ({:.1f}%), loss: {}'.format(labels.size(0), num_correct, num_correct/labels.size(0)*100, loss.item()))
                
        
        return losses, accuracy, confusion


    def _evaluate_performance( self, labels, outputs ):
        predicted = nn.Sigmoid()(outputs)
        #perf = (predicted[:,:5].argmax(1)==labels[:,:5].argmax(1))
        perf2 = (predicted[:,5:].round()==labels[:,5:])
        #print(perf2)
        num_correct = float(perf2[:,0].sum())
        #print(outputs, predicted, labels, perf, perf.sum().item(), labels.size(0))
        return num_correct

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
        
        transform_list = [
                color_jitter,
                rotation,
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=image_size, scale=(0.25,1.0), ratio=(1,1)),
                transforms.ToTensor(),
            ]
        
        if self.normalize_data:
            normalize = transforms.Normalize(mean=[0.3198, 0.1746, 0.0901],
                                            std=[0.2287, 0.1286, 0.0723])
            transform_list.append(normalize)

        train_transform = transforms.Compose(transform_list)

        self.train_dataset = EnhancedImageFolder.BinaryMultilabelCSVImageFolder(csv_file=self.csv_file,
                                                        root=self.config['files'].get('train path', './train'),
                                                        transform=train_transform, target_transform=None)
        
        self.classes = self.train_dataset.class_to_idx
        self.num_classes = len(self.classes)
    
    def _load_test_data( self ):
        image_size = self.config['files'].getint('image size', 299)
        # normalization factors for the DMR dataset were manually derived

        transform_list = [
                transforms.Resize(size=int(image_size*1.1)),
                transforms.CenterCrop(size=image_size),
                transforms.ToTensor(),
            ]
        
        if self.normalize_data:
            normalize = transforms.Normalize(mean=[0.3198, 0.1746, 0.0901],
                                            std=[0.2287, 0.1286, 0.0723])
            transform_list.append(normalize)

        test_transform = transforms.Compose(transform_list)

        self.test_dataset = EnhancedImageFolder.BinaryMultilabelCSVImageFolder(csv_file=self.csv_file,
                                                        root=self.config['files'].get('test path', './test'),
                                                        transform=test_transform, target_transform=None)
        
        self.classes = self.test_dataset.class_to_idx
        self.num_classes = len(self.classes)
