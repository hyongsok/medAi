import os
import sys
import csv
from pathlib import Path
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

def get_image_files(directory, class_index, extensions):
    images = []
    d = os.path.expanduser(directory)
    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, class_index)
                images.append(item)

    return images

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class RobustImageFolder(ImageFolder):
    """ImageFolder that samples from subdirectories and the directory itself.
    """
    def __init__(self, root, loader=default_loader, transform=None, target_transform=None, extensions=IMG_EXTENSIONS):

        try:
            super().__init__(root=root, loader=loader, transform=transform, target_transform=target_transform)
        except RuntimeError:
            self.root = root
            self.loader = loader
            self.extensions = extensions

            self.classes = []
            self.class_to_idx = dict()
            self.samples = []
            self.targets = []
            self.imgs = self.samples

            self.transform = transform
            self.target_transform = target_transform

        self.classes.append('.')
        self.class_to_idx = dict(enumerate(self.classes))
        root_images = get_image_files(root, len(self.classes)-1, extensions)
        self.samples +=  root_images
        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 files in " + root + " or its subfolders\n"
                               "Supported extensions are: " + ",".join(extensions)))
        
        self.files = [s[0] for s in self.samples]
        self.imgs = self.samples



    def _find_classes(self, directory):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        classes.append('.')
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


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

    

class BinaryMultilabelCSVImageFolder(RobustImageFolder):
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




