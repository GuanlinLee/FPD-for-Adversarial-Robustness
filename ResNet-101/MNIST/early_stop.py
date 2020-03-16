import numpy as np
import torch

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics,n,opt1,opt2,epoch):
        if self.best is None:
            self.best = metrics
            n.eval()
            checkpoint = {
                'state_dict': n.state_dict(),
                'opt_state_dict': opt1.state_dict(),
                'opt_state_dict2':opt2.state_dict(),
                'epoch': epoch
            }
            #if loss_list!=[]:
            #    for i in range(len(loss_list)):
            #        checkpoint['loss%s'%str((i+1))]=loss_list[i]
            torch.save(checkpoint, 'bestmodel_params.pkl')
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            n.eval()
            checkpoint = {
                'state_dict': n.state_dict(),
                'opt_state_dict': opt1.state_dict(),
                'opt_state_dict2':opt2.state_dict(),
                'epoch': epoch
            }
            #if loss_list!=[]:
            #    for i in range(len(loss_list)):
            #        checkpoint['loss%s'%str((i+1))]=loss_list[i]
            torch.save(checkpoint, 'bestmodel_params.pkl')
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a <= best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a >= best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a <= best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a >= best + (
                            best * min_delta / 100)