import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    if len(sys.argv) < 2:
        print("Usage: python", sys.argv[0], "evaluationfile.dat")
        return
    
    filename = sys.argv[1]

    saved = torch.load(filename)

    train_loss = saved['train_loss']
    train_accuracy = saved['test_accuracy']
    test_loss = saved['test_loss']
    test_accuracy = saved['test_accuracy']
    test_confusion = saved['test_confusion']
    if 'classes' in saved.keys():
        classes = saved['classes']
    else:
        classes =  {'mildNPDR': 0, 'modNPDR': 1, 'normal': 2, 'severe_NPDR': 3}


    plt.plot(train_loss.numpy(), 'r', label='train loss')
    plt.plot(test_loss.numpy(), 'b', label='test loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xticks(np.arange(train_loss.size(0)))
    plt.savefig('loss.png')
    plt.clf()


    plt.plot(train_accuracy.numpy(), 'r', label='train accuracy')
    plt.plot(test_accuracy.numpy(), 'b', label='test accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(train_accuracy.size(0)))
    plt.savefig('accuracy.png')
    plt.clf()


    ax = plt.subplots(1, test_confusion.size(0), True, True)
    ax[1][0].set_ylabel('predicted label')
    ax[1][0].set_yticklabels(classes.keys())
    for ii in range(test_confusion.size(0)):
        ax[1][ii].imshow(test_confusion[0,:,:].numpy(), origin='lower')
        ax[1][ii].set_xlabel('true label')
        ax[1][ii].set_xticks(np.arange(test_confusion.size(2)))
        ax[1][ii].set_xticklabels(classes.keys(), rotation=45)
        ax[1][ii].set_yticks(np.arange(test_confusion.size(2)))
    ax[0].savefig('confusion.png')
    plt.clf()


if __name__ == '__main__':
    main()