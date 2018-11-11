import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import time
from time_left import pretty_time_left, pretty_print_time
import numpy as np

import RetinaChecker
import EnhancedImageFolder
from helper_functions import AverageMeter, AccuracyMeter

class RetinaCheckerMultiClass(RetinaChecker.RetinaChecker):
    """Deep learning model
    """
   
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

            # store results & evaluate accuracy
            losses.update(loss.item(), images.size(0))

            num_correct = self._evaluate_performance( labels, outputs )

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

                num_correct = self._evaluate_performance( labels, outputs )

                top1.update(num_correct, labels.size(0))

                print('Test - samples: {}, correct: {} ({:.1f}%), loss: {}'.format(labels.size(0), num_correct, top1.avg*100, loss.item()))
                #for pred, lab in zip(predicted, groundtruth):
                #    confusion[pred, lab] += 1
        
        return losses, top1, confusion


    def _evaluate_performance( self, labels, outputs ):
        predicted = nn.Sigmoid()(outputs).round()
        print(outputs)
        print(predicted)
        print(labels)
        print(labels.size(0))
        perf = (predicted == labels)
        num_correct = (perf.sum(1)/labels.size(1)).sum().item()
        return num_correct

    def _exact_match_ratio( self, labels, predicted ):
        perf = (predicted == labels)
        num_correct = (perf.sum(1)/labels.size(1)).sum().item()
        return num_correct, labels.size(0)

    def _accuracy( self, labels, predicted ):
        matching_labels = (labels*predicted).sum(1)
        total_labels = np.sign(labels+predicted).sum(1)
        accuracy_sum = (matching_labels/total_labels).sum().item()
        return accuracy_sum, labels.size(0)

    def _precision( self, labels, predicted ):
        matching_labels = (labels*predicted).sum(1)
        correct_labels = labels.sum(1)
        precision_sum = (matching_labels/correct_labels).sum().item()
        return precision_sum, labels.size(0)

    def _recall( self, labels, predicted ):
        matching_labels = (labels*predicted).sum(1)
        predicted_labels = labels.sum(1)
        recall_sum = (matching_labels/predicted_labels).sum().item()
        return recall_sum, labels.size(0)

    def _f1_measure( self, labels, predicted ):
        matching_labels = (labels*predicted).sum(1)
        predicted_labels = labels.sum(1)
        correct_labels = labels.sum(1)
        f1_sum = (2*matching_labels/(predicted_labels + correct_labels)).sum().item()
        return f1_sum, labels.size(0)

    def _hamming_loss( self, labels, predicted ):
        missing_labels = ((labels - predicted) > 0).sum().item()
        wrong_labels = ((predicted - labels) > 0).sum().item()
        hamming_sum = (missing_labels+wrong_labels).sum().item()
        return hamming_sum, labels.size(0)*labels.size(1)

    def _initialize_model( self ):
        model_loader = None
        if self.model_name in models.__dict__.keys():
            model_loader = models.__dict__[self.model_name]
        else:
            print('Could not identify model')
            return
        
        self.model = model_loader( pretrained=self.model_pretrained, num_classes=self.num_classes, **self.model_kwargs)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential( nn.Linear(num_ftrs, self.num_classes) )
        self.model = self.model.to(self.device)


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

        self.train_dataset = EnhancedImageFolder.BinaryMultilabelImageFolder(root=self.config['files'].get('train path', './train'),
                                                        transform=train_transform, target_transform=None)
        
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

        self.test_dataset = EnhancedImageFolder.BinaryMultilabelImageFolder(root=self.config['files'].get('test path', './test'),
                                                        transform=test_transform, target_transform=None)
        
        self.classes = self.test_dataset.class_to_idx
        self.num_classes = len(self.classes)


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