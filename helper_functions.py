import numpy as np
import torch

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
