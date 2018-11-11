import os
import csv
import numpy as np
from PIL import Image
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

        self.samples, self.targets = self._convert_to_binart_labels()
        self.imgs = self.samples

    def _convert_to_binart_labels(self):
        target = []
        images = []
        binary_labels = np.eye(len(self.classes), dtype=np.int)
        for filename, class_index in self.samples:
            target.append(torch.Tensor(binary_labels[class_index]))
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

        self.samples, self.targets, self.classes = self._parse_csv(csv_file)
        self.imgs = self.samples
        

    def _parse_csv(self, csv_file):
        with open(csv_file) as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            header = next(reader)
            classes = header[1:]
            target = []
            images = []
            existing_files = [os.path.basename(s[0]) for s in self.samples]
            for row in reader:
                filename = row[0]
                if filename in existing_files:
                    images.append((filename, tuple(row[1:])))
                    target.append(torch.Tensor(row[1:]))
                else:
                    print('File {} not found in image folder. Skipping entry.')
        
        return images, target, classes




