import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    if len(sys.argv) < 2:
        print("Usage: python", sys.argv[0], "evaluationfile.dat")
        return
    
    filename = sys.argv[1]

    saved = torch.load(filename, map_location='cpu')

    train_loss = saved['train_loss']
    train_accuracy = saved['test_accuracy']
    test_loss = saved['test_loss']
    test_accuracy = saved['test_accuracy']
    test_confusion = saved['test_confusion']
    if 'classes' in saved.keys():
        classes = saved['classes']
    else:
        classes =  {'mildNPDR': 0, 'modNPDR': 1, 'normal': 2, 'severe_NPDR': 3}

    plot_loss( train_loss, test_loss )
    plot_accuracy( train_accuracy, test_accuracy )
    plot_confusion( test_confusion, classes )
    plot_2_class_evaluation( test_accuracy, test_confusion )

def plot_loss( train_loss, test_loss ):
    plt.plot(train_loss.numpy(), 'r', label='train loss')
    plt.plot(test_loss.numpy(), 'b', label='test loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xticks(np.arange(0, train_loss.size(0), int(train_loss.size(0)/10)))
    plt.savefig('loss.png')
    plt.clf()

def plot_accuracy( train_accuracy, test_accuracy ):
    plt.plot(train_accuracy.numpy(), 'r', label='train accuracy')
    plt.plot(test_accuracy.numpy(), 'b', label='test accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(0, test_accuracy.size(0), int(test_accuracy.size(0)/10)))
    plt.savefig('accuracy.png')
    plt.clf()

def plot_confusion( test_confusion, classes ):
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

def plot_2_class_evaluation( test_accuracy, test_confusion ):
    specificity = np.zeros_like(test_accuracy)
    sensitivity = np.zeros_like(test_accuracy)
    accuracy = np.zeros_like(test_accuracy)
    for ii in range(specificity.size):
        confusion_2class = np.vstack((np.vstack((test_confusion[ii,:,(1,3)].sum(1),
                                            test_confusion[ii,:,(0,2,4)].sum(1)))[:,(1,3)].sum(1), 
                                np.vstack((test_confusion[ii,:,(1,3)].sum(1),
                                            test_confusion[ii,:,(0,2,4)].sum(1)))[:,(0,2,4)].sum(1))
                                )
        accuracy[ii] = np.diag(confusion_2class).sum()/confusion_2class.sum()
        sensitivity[ii] = confusion_2class[1,1]/confusion_2class[:,1].sum()
        specificity[ii] = confusion_2class[0,0]/confusion_2class[:,0].sum()

    max_acc = np.max(accuracy)
    best_acc = np.argmax(accuracy)
    max_sen = np.max(sensitivity)
    best_sen = np.argmax(sensitivity)
    max_spe = np.max(specificity)
    best_spe = np.argmax(specificity)


    ax = plt.subplots(1, 3, True, True, figsize=(20,3))

    ax[1][0].plot(accuracy, 'r', label='accuracy')
    ax[1][0].plot(best_acc, max_acc, 'ro')
    ax[1][0].annotate(
            'epoch {} best acc @ {:.1f}%'.format(best_acc, max_acc*100),
            xy=(best_acc, max_acc), xytext=(20, 20),
            textcoords='offset points', ha='right', va='bottom',
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    ax[1][0].legend(loc='lower right')
    ax[1][0].set_ylabel('accuracy')
    ax[1][0].set_xlabel('epochs')
    ax[1][0].set_xticks(np.arange(0,accuracy.size+1, 10))

    ax[1][1].plot(sensitivity, 'b', label='sensitivity')
    ax[1][1].plot(best_sen, max_sen, 'bo')
    ax[1][1].annotate(
            'epoch {} best sen @ {:.1f}%'.format(best_sen, max_sen*100),
            xy=(best_sen, max_sen), xytext=(20, 20),
            textcoords='offset points', ha='right', va='bottom',
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    ax[1][1].legend(loc='lower right')
    ax[1][1].set_xlabel('epochs')
    ax[1][1].set_ylabel('sensitivity')
    ax[1][1].set_xticks(np.arange(0,accuracy.size+1, 10))

    ax[1][2].plot(specificity, 'k', label='specificity')
    ax[1][2].plot(best_spe, max_spe, 'ko')
    ax[1][2].annotate(
            'epoch {} best spec @ {:.1f}%'.format(best_spe, max_spe*100),
            xy=(best_spe, max_spe), xytext=(20, 20),
            textcoords='offset points', ha='right', va='bottom',
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    ax[1][2].legend(loc='lower right')
    ax[1][2].set_xticks(np.arange(0,accuracy.size+1, 10))
    ax[1][2].set_ylabel('specificity')
    ax[1][2].set_xlabel('epochs')
    plt.savefig('2class.png')

if __name__ == '__main__':
    main()