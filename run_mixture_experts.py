import sys
import json
import configparser
import numpy as np
import torch
from RetinaChecker import RetinaChecker
from helper_functions import reduce_to_2_classes

def validate( experts ):
    '''Evaluates the model given the criterion and the data in test_loader
    
    Arguments:
        test_loader {torch.utils.data.DataLoader} -- contains the data for training, if None takes internal test_loader     
    
    Returns:
        AverageMeter -- training loss
        AccuracyMeter -- training accuracy
        numpy.Array -- [num_classes, num_classes] confusion matrix, columns are true classes, rows predictions
    '''

    # assuming that all models have the same test loader. Or at least all models will be compared to the one test loader
    test_loader = experts[0].test_loader
    all_labels = np.array(experts[0].test_loader.dataset.samples)[:,1].astype(np.int)
    all_predictions = np.zeros((len(experts), all_labels.size), dtype=np.int)
    all_outputs = np.zeros((len(experts), all_labels.size, experts[0].model.fc.out_features), dtype=np.float)

    for expert_number, expert in enumerate(experts):
        expert.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():

            counter = 0
            for images, labels in test_loader:
                images = images.to(expert.device)
                labels = labels.to(expert.device)

                outputs = expert.model(images)

                _, predicted = torch.max(outputs.data, 1)
                all_predictions[expert_number, counter:counter+len(predicted)] = predicted.data
                all_outputs[expert_number, counter:counter+len(predicted), :] = outputs.data
                counter += len(predicted)
    
    predicted = np.zeros_like(all_labels)
    for i in range(all_labels.size):
        classes, counts = np.unique(all_predictions[:,i], return_counts=True)
        predicted[i] = classes[np.argmax(counts)]
    top1 = (predicted == all_labels)

    confusion = torch.zeros((experts[0].num_classes, experts[0].num_classes), dtype=torch.float)
    for pred, lab in zip(predicted, all_labels):
        confusion[pred, lab] += 1
    
    np.savez('temp.npz', all_labels=all_labels, all_predictions=all_predictions, all_outputs=all_outputs, predicted=predicted, confusion=confusion, classes=experts[0].classes)


    return top1, confusion

def main():


    # Reading configuration file
    config = configparser.ConfigParser()
    checkpoints = None
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            checkpoints = json.load(f)
        config.read(sys.argv[1])
    else:
        print('Usage: python run_mixture_experts experts.cfg experts.json')
        return

    experts = []
    for checkpoint in checkpoints:
        config['input']['checkpoint'] = checkpoint
        rc = RetinaChecker()
        rc.initialize( config )
   
        rc.load_state()
        experts.append(rc)

    top1, confusion = validate(experts)

    print('Test Accuracy of the model on the {} test images: {} %'.format(top1.size, top1.sum()/top1.size*100))
    print('Classes: {}'.format(rc.classes))
    print('Confusion matrix:\n', (confusion))

    confusion_2class = reduce_to_2_classes( confusion, [(1,3), (0,2,4)])
    print('Accuracy: {:.1f}%'.format(np.diag(confusion_2class).sum()/confusion_2class.sum()*100))
    print(confusion_2class)
    print('Sensitivity: {:.1f}%'.format(confusion_2class[1,1]/confusion_2class[:,1].sum()*100))
    print('Specificity: {:.1f}%'.format(confusion_2class[0,0]/confusion_2class[:,0].sum()*100))


if __name__ == '__main__':
    main()