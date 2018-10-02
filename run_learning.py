import sys, os, glob
import configparser
import time
import numpy as np
import torch
from time_left import pretty_time_left, pretty_print_time
from RetinaChecker import RetinaChecker

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


def main():

    # Reading configuration file
    config = configparser.ConfigParser()
    if len(sys.argv) > 1:
        config.read(sys.argv[1])
    else:
        config.read('default.cfg')

    rc = RetinaChecker()
    rc.initialize( config )

    # loading previous models
    if config['input'].getboolean('resume', False):
        rc.load_state()

    print('Model set up. Ready to train.')

    # Patch statistics requires the data loader to load all patches once
    # to analyze the data. This takes a lot of time. Use only when you
    # change something on the sampler to see the results. Should be False
    # for regular learning passes.
    if config['files'].getboolean('show patch stats', False):
        print_dataset_stats( rc.train_dataset, rc.train_loader )
        print_dataset_stats( rc.test_dataset, rc.test_loader )
    print('Data loaded. Setting up model.')
    

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