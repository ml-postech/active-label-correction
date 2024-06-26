import os
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import DiceScore
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string
    
    def to_save(self, results, save_name):
        with open(os.path.join('./results', save_name + '.txt'), 'w') as f:
            for k, v in results.items():
                if k != "Class IoU":
                    f.write(str(k) + " " + str(v) + '\n')
                else:
                    f.write("Class IoU" + '\n')
                    for key, value in results["Class IoU"].items():
                        f.write(str(value) + '\n')
        
    def _fast_hist(self, label_true, label_pred):
        # mask = (label_true >= 0) & (label_true < self.n_classes)
        
        # for Data mIoU
        mask = (label_true >= 0) & (label_true < self.n_classes) & (label_pred >= 0) & (label_pred < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class BinarySegMetrics(_StreamMetrics):
    """
    Segmentation metric for binary class
    """
    def __init__(self):
        self.n_classes = 2
        self.get_dicescore = DiceScore()
        ### metircs: dice, IoU, precision, recall
        self.dice_score = 0
        self.dice = 0
        self.IoU = 0
        self.precision = 0
        self.recall = 0
        self.nsample = 0

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            # self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
            self.dice_score += self.get_dicescore(lt, lp)
            lt = lt.flatten()
            lp = lp.flatten()
            self.dice += (f1_score(lt, lp))
            self.IoU += (jaccard_score(lt, lp))
            self.precision += (precision_score(lt, lp))
            self.recall += (recall_score(lt, lp))
            self.nsample += 1

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string
    
    def to_save(self, results, save_name):
        with open(os.path.join('./results', save_name + '.txt'), 'w') as f:
            for k, v in results.items():
                if k != "Class IoU":
                    f.write(str(k) + " " + str(v) + '\n')
                else:
                    f.write("Class IoU" + '\n')
                    for key, value in results["Class IoU"].items():
                        f.write(str(value) + '\n')

    def get_results(self):
        """Returns accuracy score evaluation result.
            - dice_score
            - dice
            - IoU
            - precision
            - recall
            - nsample
        """

        return {
                "dice_score": self.dice_score / self.nsample,
                "dice": self.dice / self.nsample,
                "Mean IoU": self.IoU / self.nsample,
                "precision": self.precision / self.nsample,
                "recall": self.recall / self.nsample,
                "nsample": self.nsample,
            }
        
    def reset(self):
        self.dice_score = 0
        self.dice = 0
        self.IoU = 0
        self.precision = 0
        self.recall = 0
        self.nsample = 0

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
