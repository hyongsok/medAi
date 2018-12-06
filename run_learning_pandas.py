import sys, os, glob
import configparser
import time
import numpy as np
import torch
from time_left import pretty_time_left, pretty_print_time
from RetinaCheckerPandas import RetinaCheckerPandas
from helper_functions import initialize_meters, save_performance
from make_default_config import get_config

def main():

    # Reading configuration file
    config = configparser.ConfigParser()
    if len(sys.argv) > 1:
        config.read(sys.argv[1])
    else:
        config = get_config()

    # create the checker class and initialize internal variables
    rc = RetinaCheckerPandas()
    rc.initialize(config)

    rc.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Loading data sets based on configuration and enable normaization
    rc.load_datasets()

    # Initializing sampler and data (=patch) loader
    rc.create_dataloader(config['files'].getint('num workers', 0))

    # Initialize the model
    rc.initialize_model()
    rc.initialize_criterion()
    rc.initialize_optimizer()

    if config['input'].getboolean('evaluation only', False):
        print('Configuration file has evaluation only flag set. This is the learning script. Please adjust configuration or use the appropriate script.')
        return

    # loading previous models
    if config['input'].getboolean('resume', False):
        rc.load_state()

    print('Model set up. Ready to train.')
    print(rc)

    # Performance meters initalized (either empty or from file)
    num_epochs = rc.start_epoch + config['hyperparameter'].getint('epochs', 10)
    train_loss, train_accuracy, test_loss, test_accuracy, test_confusion = initialize_meters( config, rc.start_epoch, num_epochs, 2 )

    # Starting training & evaluation
    start_time = time.time()
    best_accuracy = 0
    best_accuracy_2class = 0
    best_sensitivity = 0
    best_specificity = 0
    for epoch in range(rc.start_epoch, num_epochs):
        start_time_epoch = time.time()

        # Train the model and record training loss & accuracy
        losses, accuracy = rc.train()

        # Logging and performance 
        train_loss[epoch] = losses.avg
        train_accuracy[epoch] = accuracy.avg
        
        # Validation
        losses, accuracy, confusion = rc.validate()

        # Logging and performance measuring overhead
        test_loss[epoch] = losses.avg
        test_accuracy[epoch] = accuracy.avg
        test_confusion[epoch,:,:] = confusion

        #print('Test Accuracy of the model on the {} test images: {} %'.format(accuracy.count, accuracy.avg*100))
        #print('Classes: {}'.format(rc.classes))
        print('Confusion matrix:\n', (confusion))

        #confusion_2class = reduce_to_2_classes( confusion, [(0,1), (2,3,4)])
        confusion_2class = confusion
        accuracy_2class = np.diag(confusion_2class).sum()/confusion_2class.sum()
        sensitivity_2class = confusion_2class[1,1]/confusion_2class[:,1].sum()
        specificity_2class = confusion_2class[0,0]/confusion_2class[:,0].sum()
        print('Accuracy: {:.1f}%'.format(accuracy_2class*100))
        print('Sensitivity: {:.1f}%'.format(sensitivity_2class*100))
        print('Specificity: {:.1f}%'.format(specificity_2class*100))

        if config['output'].getboolean('save during training', False) and ((epoch+1) % config['output'].getint('save every nth epoch', 10) == 0):
            rc.save_state( train_loss[:(epoch+1)], 
                        train_accuracy[:(epoch+1)], test_loss[:(epoch+1)], 
                        test_accuracy[:(epoch+1)], test_confusion[:(epoch+1),:,:], 
                        config['output'].get('filename', 'model')+'_after_epoch_{}'.format(epoch+1)+config['output'].get('extension', '.ckpt') )
        
        if accuracy.avg > best_accuracy:
            rc.save_state( train_loss[:(epoch+1)], 
                        train_accuracy[:(epoch+1)], test_loss[:(epoch+1)], 
                        test_accuracy[:(epoch+1)], test_confusion[:(epoch+1),:,:], 
                        config['output'].get('filename', 'model')+'_best_accuracy'+config['output'].get('extension', '.ckpt') )
            best_accuracy = accuracy.avg

        if accuracy.avg >= best_accuracy:
            rc.save_state( train_loss[:(epoch+1)], 
                        train_accuracy[:(epoch+1)], test_loss[:(epoch+1)], 
                        test_accuracy[:(epoch+1)], test_confusion[:(epoch+1),:,:], 
                        config['output'].get('filename', 'model')+'_best_accuracy_later'+config['output'].get('extension', '.ckpt') )
            best_accuracy = accuracy.avg
        
        if sensitivity_2class > best_sensitivity:
            rc.save_state( train_loss[:(epoch+1)], 
                        train_accuracy[:(epoch+1)], test_loss[:(epoch+1)], 
                        test_accuracy[:(epoch+1)], test_confusion[:(epoch+1),:,:], 
                        config['output'].get('filename', 'model')+'_best_sensitivity'+config['output'].get('extension', '.ckpt') )
            best_sensitivity = sensitivity_2class

        if specificity_2class > best_specificity:
            rc.save_state( train_loss[:(epoch+1)], 
                        train_accuracy[:(epoch+1)], test_loss[:(epoch+1)], 
                        test_accuracy[:(epoch+1)], test_confusion[:(epoch+1),:,:], 
                        config['output'].get('filename', 'model')+'_best_specificity'+config['output'].get('extension', '.ckpt') )
            best_specificity = specificity_2class

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
    if config['output'].getboolean('cleanup', True):
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