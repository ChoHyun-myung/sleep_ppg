import matplotlib.pyplot as plt

from db import *
from dbloader import *
from torchvision import transforms
from model import *
from train import *

from sklearn.metrics import *


def label_dist():
    record_list = np.loadtxt('record_list', dtype='str', delimiter=',')
    dbset = GamerPPG(record_list, './db/',
                     pre_processing=transforms.Compose([
                         Normalization(), ToTensor()
                     ]))
    loader = Loader(dbset, batch_size=32, shuffle=True, valid_split=0, seed=4, worker=1)
    loader = loader.fetch()

    label = np.zeros(7)
    for x, y in loader:
        y = y.to('cpu').numpy()
        for y_ in y:
            label[y_] += 1
    print(label)


def exp_list(sub=None):
    if sub is None:
        tr_list = np.loadtxt('train', dtype='str', delimiter=',')
        val_list = np.loadtxt('val', dtype='str', delimiter=',')
        tr_set = GamerPPG(tr_list, './db/',
                          pre_processing=transforms.Compose([
                              Normalization(), ToTensor()
                          ]))
        val_set = GamerPPG(val_list, './db/',
                           pre_processing=transforms.Compose([
                               Normalization(), ToTensor()
                           ]))
        ld_tr = Loader(tr_set, batch_size=32, shuffle=True, valid_split=0, seed=4, worker=1)
        ld_val = Loader(val_set, batch_size=32, shuffle=True, valid_split=0, seed=4, worker=1)
        loader_tr = ld_tr.fetch()
        loader_val = ld_val.fetch()
    else:
        record_list = [x for x in np.loadtxt('record_list', dtype='str', delimiter=',') if sub in x]

        dbset = GamerPPG(record_list, './db/',
                         pre_processing=transforms.Compose([
                             Normalization(), ToTensor()
                         ]))

        loader = Loader(dbset, batch_size=32, shuffle=True, valid_split=.2, seed=4, worker=1)
        loader_tr, loader_val = loader.fetch()

    return loader_tr, loader_val


def cv_list(sub):
    print(f'cross validation {sub} vs. others')
    tr_list = [x for x in np.loadtxt('record_list', dtype='str', delimiter=',') if sub in x]
    val_list = [x for x in np.loadtxt('record_list', dtype='str', delimiter=',') if ~(sub in x)]

    tr_set = GamerPPG(tr_list, './db/',
                      pre_processing=transforms.Compose([
                          Normalization(), ToTensor()
                      ]))
    val_set = GamerPPG(val_list, './db/',
                       pre_processing=transforms.Compose([
                           Normalization(), ToTensor()
                       ]))
    ld_tr = Loader(tr_set, batch_size=32, shuffle=True, valid_split=0, seed=4, worker=1)
    ld_val = Loader(val_set, batch_size=32, shuffle=True, valid_split=0, seed=4, worker=1)
    loader_tr = ld_tr.fetch()
    loader_val = ld_val.fetch()
    return loader_tr, loader_val


def train_fn(tr, val, result_name):
    loader_tr = WrappedDataLoader(tr,
                                  x_dim=[1, window * 100],
                                  y_dim=1)
    loader_val = WrappedDataLoader(val,
                                   x_dim=[1, window * 100],
                                   y_dim=1)
    model = ModelSleepiness(input_shape=[1, window * 100],
                            output_shape=7,
                            n_blocks=5,
                            init_channel=8,
                            kernel_size=15,
                            dilation=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = torch.nn.CrossEntropyLoss()  # Including Sigmoid function

    fit_fn = Fit(model=model, loss_fn=loss_fn,
                 optimizer=optimizer)

    fit_fn.fit(epochs=20,
               dl_train=loader_tr, dl_valid=loader_val)

    cm = confusion_matrix(y_true=fit_fn.best_label, y_pred=fit_fn.best_pred,
                          labels=np.arange(0, 7, 1) + 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'./results/{result_name}.png')


if __name__ == '__main__':
    label_dist()
    window = 300

    for sub in ['gamer1', 'gamer2', 'gamer3', 'gamer4', 'gamer5']:
        print(sub)
        loader_tr, loader_val = exp_list(sub)
        train_fn(loader_tr, loader_val, result_name=f'{sub}')
        loader_tr, loader_val = cv_list(sub)
        train_fn(loader_tr, loader_val, result_name=f'{sub}_cv')
    loader_tr, loader_val = exp_list(None)
    train_fn(loader_tr, loader_val, result_name='all')
