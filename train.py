import numpy as np
import sklearn.metrics
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import *


class Fit:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.min_loss = np.inf
        self.best_acc = 0
        self.best_pred, self.best_label = 0, 0

        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(dev)

    def loss_batch(self, xb, yb, optimizer=None, lb=None):
        prediction = self.model(xb)

        loss = self.loss_fn(prediction, yb)
        if optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        _prediction = prediction.data.cpu().numpy()
        _yb = yb.data.cpu().numpy()
        return loss.item(), len(xb), _prediction, _yb

    def fit(self, epochs, dl_train, dl_valid):
        for epoch in range(1, epochs + 1):
            # Training
            self.train_fn(dl_train, epoch, mode='train')

            # Validation
            self.train_fn(dl_valid, epoch, mode='eval')

        print(f'Training finished, best acc {self.best_acc:.4f}')
        return self.min_loss

    def train_fn(self, dl, epoch, mode='train'):
        if mode == 'train':
            self.model.train()
            losses, nums, predictions, ybs = zip(
                *[self.loss_batch(xb, yb, optimizer=self.optimizer) for xb, yb in dl]
            )
        elif mode == 'eval':
            self.model.eval()
            with torch.no_grad():
                losses, nums, predictions, ybs = zip(
                    *[self.loss_batch(xb, yb) for xb, yb in dl]
                )
        loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # Converting torch tensor to numpy
        predictions, ybs = list(predictions), list(ybs)
        y_pred = np.array(predictions[:-1]).reshape([-1, 7])
        y_pred = np.append(y_pred, predictions[-1].reshape(-1, 7), axis=0)  # [None, classes]
        y_label = np.array(ybs[:-1]).reshape(-1)
        y_label = np.append(y_label, ybs[-1].reshape(-1), axis=0)

        def softmax(x):
            max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
            e_x = np.exp(x - max)  # subtracts each row with its max value
            sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
            f_x = e_x / sum
            return f_x
        y_pred_softmax = softmax(y_pred)
        y_pred_argmax = np.argmax(y_pred_softmax, axis=1)

        acc = np.sum(y_pred_argmax == y_label) / len(y_label)

        if mode == 'train':
            # print(f'Epoch[{epoch}]: Training   loss={loss:.5f}, acc={acc:.4f}')
            pass
        elif mode == 'eval':
            if self.best_acc < acc:
                self.best_acc = acc
                self.best_pred = y_pred_argmax
                self.best_label = y_label

            # print(f'Epoch[{epoch}]: Validation loss={loss:.5f}, acc={acc:.4f}\n')
            # for cls in classes:
            #     print(f'{cls} / F1 score: {metric.score[cls]["f1_score"]:.3f}')
