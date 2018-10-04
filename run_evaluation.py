import sys
import configparser
import numpy as np
from RetinaChecker import RetinaChecker
from helper_functions import reduce_to_2_classes

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
    if not config['input'].getboolean('resume', False):
        print('Please set the resume parameter to True')
        return
    
    rc.load_state()

    _, top1, confusion = rc.validate()

    print('Test Accuracy of the model on the {} test images: {} %'.format(top1.count, top1.avg*100))
    print('Classes: {}'.format(rc.classes))
    print('Confusion matrix:\n', (confusion))

    confusion_2class = reduce_to_2_classes( confusion, [(1,3), (0,2,4)])
    print('Accuracy: {:.1f}%'.format(np.diag(confusion_2class).sum()/confusion_2class.sum()*100))
    print(confusion_2class)
    print('Sensitivity: {:.1f}%'.format(confusion_2class[1,1]/confusion_2class[:,1].sum()*100))
    print('Specificity: {:.1f}%'.format(confusion_2class[0,0]/confusion_2class[:,0].sum()*100))


if __name__ == '__main__':
    main()