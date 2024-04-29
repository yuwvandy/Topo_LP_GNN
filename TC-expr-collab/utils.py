import numpy as np
import random
import torch
import os
from torch_sparse import SparseTensor
from torch import Tensor
import torch_sparse
from typing import List, Tuple

np.set_printoptions(precision=4)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def path_everything(dataset, model):
    if not os.path.exists('./data/' + dataset):
        os.mkdir('./data/' + dataset)

    if not os.path.exists('./res/' + dataset):
        os.mkdir('./res/' + dataset)
    if not os.path.exists('./res/' + dataset + '/' + model):
        os.mkdir('./res/' + dataset + '/' + model)

    if not os.path.exists('./model/' + dataset):
        os.mkdir('./model/' + dataset)
    if not os.path.exists('./model/' + dataset + '/' + model):
        os.mkdir('./model/' + dataset + '/' + model)
        

def batch_to_gpu(batch, device):
    for c in batch:
        batch[c] = batch[c].to(device)

    return batch


def elem2spm(element: Tensor, sizes: List[int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    # print(spm, spm.storage.row(), spm.storage.col())
    # print(torch.bitwise_left_shift(spm.storage.row(), 32))
    # print(torch.bitwise_left_shift(spm.storage.row(), 32).add_(spm.storage.col()))
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem


def spmoverlap_(adj1, adj2):
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1

    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    retelem = element2[mask]


    return elem2spm(retelem, adj1.sizes())


def adjoverlap(adj1, adj2, edge):

    # a wrapper for functions above.
    adj1 = adj1[edge[:, 0]]
    adj2 = adj2[edge[:, 1]]

    adjoverlap = spmoverlap_(adj1, adj2) #edges * nodes

    return adjoverlap



def getLabel(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype('float')




def test_one_batch_group(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]

    r = getLabel(groundTrue, sorted_items)

    pres, recalls, ndcgs, hit_ratios, F1s = [], [], [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        ndcgs.append(NDCGatK_r(groundTrue, r, k))
        hit_ratios.append(Hit_at_k(r, k))
        recalls.append(ret['Recall'])
        pres.append(ret['Precision'])

        temp = ret['Precision'] + ret['Recall']
        temp[temp == 0] = float('inf')
        F1s.append(2 * ret['Precision'] * ret['Recall'] / temp)
        # F1s[np.isnan(F1s)] = 0

    return np.stack(recalls).transpose(1, 0), np.stack(ndcgs).transpose(1, 0), np.stack(hit_ratios).transpose(1, 0), np.stack(pres).transpose(1, 0), np.stack(F1s).transpose(1, 0)




def Hit_at_k(r, k):
    right_pred = r[:, :k].sum(axis=1)

    return 1. * (right_pred > 0)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    # print(right_pred, 2213123213)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    recall[np.isnan(recall)] = 0
    precis = right_pred / precis_n
    return {'Recall': recall, 'Precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """

    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix

    # print(max_r[0], pred_data[0])
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)

    idcg[idcg == 0.] = 1.  # it is OK since when idcg == 0, dcg == 0
    ndcg = dcg / idcg
    # ndcg[np.isnan(ndcg)] = 0.

    return ndcg



def elem2spm(element: Tensor, sizes: List[int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    # print(spm, spm.storage.row(), spm.storage.col())
    # print(torch.bitwise_left_shift(spm.storage.row(), 32))
    # print(torch.bitwise_left_shift(spm.storage.row(), 32).add_(spm.storage.col()))
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem


def spmoverlap_(adj1, adj2):
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1

    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    retelem = element2[mask]


    return elem2spm(retelem, adj1.sizes())


def adjoverlap(adj1, adj2, edge):

    # a wrapper for functions above.
    adj1 = adj1[edge[:, 0]]
    adj2 = adj2[edge[:, 1]]

    adjoverlap = spmoverlap_(adj1, adj2) #edges * nodes

    return adjoverlap

