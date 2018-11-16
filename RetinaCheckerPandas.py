import configparser
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
from time_left import pretty_time_left, pretty_print_time
import numpy as np

import RetinaCheckerCSV
import PandasDataset
from helper_functions import AverageMeter, AccuracyMeter

class RetinaCheckerPandas(RetinaCheckerCSV.RetinaCheckerCSV):
    """Deep learning model
    """

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

        self.train_dataset = PandasDataset.PandasDataset(source=self.csv_file, mode='csv'
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

        self.test_dataset = PandasDataset.PandasDataset(source=self.csv_file, mode='csv'
                                                        root=self.config['files'].get('test path', './test'),
                                                        transform=test_transform, target_transform=None)
        
        self.classes = self.test_dataset.class_to_idx
        self.num_classes = len(self.classes)
