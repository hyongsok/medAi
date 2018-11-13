import os
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import sklearn.preprocessing
import torch
from torchvision.datasets import ImageFolder


# loader methods stolen from torchvision/datasets/folder.py 
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)

class BinaryMultilabelImageFolder(ImageFolder):
    """BinaryMultilabelImageFolder extends the torchvision ImageFolder by taking
    a csv file (and optionally an image folder if the csv is not in the same location)
    and loads all images from the folder with class labels defined in the csv file. Expects
    binary labels, i.e. one label per column and only 0 or 1.
    Images not present in the csv file will not be loaded.
    
        Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        
        super().__init__(root=root, transform=transform, target_transform=target_transform, loader=loader)

        self.samples, self.targets = self._convert_to_binary_labels()
        self.imgs = self.samples

    def _convert_to_binary_labels(self):
        target = []
        images = []
        binary_labels = sklearn.preprocessing.label_binarize([y[1] for y in self.samples], range(len(self.classes)))
        for ii, (filename, _) in enumerate(self.samples):
            target.append(torch.Tensor(binary_labels[ii]))
            images.append((filename, target[-1]))
        
        return images, target

    

class BinaryMultilabelCSVImageFolder(ImageFolder):
    """BinaryMultilabelCSVImageFolder extends the torchvision ImageFolder by taking
    a csv file (and optionally an image folder if the csv is not in the same location)
    and loads all images from the folder with class labels defined in the csv file. Expects
    binary labels, i.e. one label per column and only 0 or 1.
    Images not present in the csv file will not be loaded.
    
        Args:
        csv_file (string): csv file containing all images and their classes
        root (string, optional): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, csv_file, root=None, transform=None, target_transform=None,
                 loader=default_loader):
        
        if root is None:
            root = os.path.dirname(os.path.abspath(csv_file))

        super().__init__(root=root, loader=loader, transform=transform, target_transform=target_transform)

        self.samples, self.targets, self.classes = self._parse_csv(csv_file, root)
        self.imgs = self.samples
        self.class_to_idx = dict(enumerate(self.classes))
        

    def _parse_csv(self, csv_file, root):
        with open(csv_file) as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            header = next(reader)
            classes = header[1:]
            target = []
            images = []
            root_path = Path(root)
            for row in reader:
                filename = root_path / Path(row[0])
                if filename.exists():
                    target.append(torch.Tensor([int(r) for r in row[1:]]))
                    images.append((filename, target[-1]))
                else:
                    print('File {} not found in folder {}. Skipping entry.'.format(filename, root))
            if len(target) == 0:
                raise ValueError('Could not read any file. Did you forget the slash or backslash at the end of the root (=train & test) path?')
        
        return images, target, classes




