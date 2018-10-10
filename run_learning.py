import sys, os, glob
import configparser
import time
import numpy as np
from time_left import pretty_time_left, pretty_print_time
from RetinaChecker import RetinaChecker
from helper_functions import print_dataset_stats, initialize_meters, reduce_to_2_classes, save_performance

def main():

    # Reading configuration file
    config = configparser.ConfigParser()
    if len(sys.argv) > 1:
        config.read(sys.argv[1])
    else:
        config.read('default.cfg')

    rc = RetinaChecker()
    rc.initialize( config )

    if config['input'].getboolean('evaluation only', False):
        print('Configuration if for evaluation only. This is the learning script. Please adjust configuration.')
        return

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
        losses, top1 = rc.train()
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