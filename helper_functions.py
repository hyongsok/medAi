"""Helper functions for RetinaChecker

Author: Philipp Lies (phil@lies.io)
partially derived from imagenet example
"""

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