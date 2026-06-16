from __future__ import print_function
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import seaborn as sns


to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_device(device_number):
    use_cuda = torch.cuda.is_available()
    # print(use_cuda)
    if use_cuda and device_number == '1':
        device = torch.device("cuda:1" if use_cuda else "cpu")
        # print(device)
    elif use_cuda and device_number == '0':
        device = torch.device("cuda:0" if use_cuda else "cpu")
        # print(device)
    return device

def get_feature(model, data_loader, device, normalizetion=True):
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            feature_list = []
            label_true = []
            for idx, (input, label) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                features = model(input)
                features = features.flatten().cpu()
                if normalizetion:
                    # norms = np.linalg.norm(features, axis=1, keepdims=True)
                    norms = np.linalg.norm(features, keepdims=True)
                    features = features / norms
                    feature_list.append(features)  # append flatten features
                else:
                    feature_list.append(features)  # append flatten features

                label_true.append(label.flatten())

    features_array = np.vstack(feature_list)

    return features_array

def get_ood_score_multiprototypes(feature_list, center_list, emd=False):
    min_list = []
    for feature in feature_list:
        dis_list = []
        for center in center_list:
            if emd:
                distance = wasserstein_distance(center, feature)
            else:
                difference = feature - center
                distance = np.linalg.norm(difference)
            dis_list.append(distance)
        min_dis = min(dis_list)
        min_list.append(min_dis)
    return -1*np.array(min_list)

def get_ood_gradnorm(net, loader, device, T):
    _score = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    with tqdm(loader, total=len(loader)) as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(device)
            net.zero_grad()
            output = net(data)
            num_classes = output.shape[-1]
            targets = torch.ones((data.shape[0], num_classes)).to(device)
            output = output / T
            loss = torch.mean(torch.sum(-targets * logsoftmax(output), dim=-1))

            loss.backward()
            layer_grad = net.fc.weight.grad.data

            layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
            # all_score = -layer_grad_norm
            all_score = layer_grad_norm
            _score.append(all_score)
    return np.array(_score)

def data_test_logit(data_loader, model, model_f, device, method='MaxLogit'):
    """One epoch validation"""

    # switch to evaluate mode
    model.eval()
    true_label_list = []
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            _score1 = []
            _score2 = []
            _probability = []
            _entropy = []
            for idx, (input, true_label) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                    true_label = true_label.to(device)

                true_label_list.append(true_label.cpu())
                # compute output
                logits = model(input)#logits
                probability = F.softmax((logits), dim=1)
                features = model_f(input)
                all_score1 = np.max(to_np(logits), axis=1)#maxlogit
                all_score2 = features.norm(2, dim=1).cpu().numpy()#MaxNorm
                probability_batch = np.max(to_np(probability), axis=1)
                entropy = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
                for i in range(true_label.shape[0]):
                    _score2.append(all_score2[i][0][0])
                    _score1.append(all_score1[i])
                    _probability.append(probability_batch[i])
                    _entropy.append(entropy[i])
    if method == 'MaxLogit':
        return np.array(_score1)
    elif method == 'DML':
        score = np.array(_score1) + np.array(_score2)
        # score = list(score)
        return score
    elif method == 'MaxNorm':
        return np.array(_score2)
    elif method == 'MSP':
        return np.array(_probability)
    elif method == 'Energy':
        return np.array(_entropy)

def get_ood_scores_odin(loader, net, T, noise, device):
    _score = []
    net.eval()
    with tqdm(loader, total=len(loader)) as pbar:
        for batch_idx, examples in enumerate(pbar):
            data, target = examples[0], examples[1]
            data = data.to(device)
            data = Variable(data, requires_grad=True)

            output = net(data)
            odin_score = ODIN(data, output, net, T, noise, device)
            # _score.append(-np.max(odin_score, 1))
            _score.append(np.max(odin_score, 1))

    return concat(_score).copy()

def ODIN(inputs, outputs, model, temper, noiseMagnitude1, device):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()
    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    # Using temperature scaling
    outputs = outputs / temper
    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()
    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
    gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
    gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)
    # gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    # gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    # gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    # tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    tempInputs = torch.add(inputs.data, gradient, alpha=-noiseMagnitude1)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs

def plot_distribution(value_list, label_list, savepath='nll', alpha=0.5):
    sns.set(style="white", palette="muted")
    # palette = ['#A8BAE3', '#55AB83']
    palette = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'pink']
    dict_value = {label_list[i]: value_list[i] for i in range(len(label_list))}
    sns.displot(dict_value, kind="kde", palette=palette, fill=True, alpha=alpha)
    plt.savefig(savepath, dpi=300)

def plot_values_hist(value_list, label_list, savepath='nll'):
    plt.clf() # clear the contex of figure
    plt.figure(figsize=(14/2.5, 14/2.5))
    color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'pink']
    # f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    for index, probability_list in enumerate(value_list):
        probability_array = np.array(probability_list)
        # density=True设置为密度刻度
        n1, bins1, patches1 = plt.hist(probability_array, density=True, histtype='step', bins=100, color=color_list[index],
            label=label_list[index], alpha=0.8, rwidth=0.1, linewidth=4.0)

    plt.legend()
    if savepath != 'nll':
        plt.savefig(savepath, dpi=300)


def get_ood_score(data_loader, model, model_f, device, method='DMP', centroids=None, train_loader=None, normalize_value=None):

    if method in ['MaxLogit', 'DML', 'MaxNorm', 'MSP', 'Energy']:
        score = data_test_logit(data_loader, model, model_f, device, method)
        if normalize_value!= None:
            score = score / normalize_value
    elif method == 'GradNorm':
        score = get_ood_gradnorm(model, data_loader, device, T=1)
    elif method == 'ODIN':
        score = get_ood_scores_odin(data_loader, model, T=1000, noise=0.001, device=device)

        # score = score.tolist()
    elif method in ['DMPC', 'DMPL']:
        feature = get_feature(model_f, data_loader, device)
        score = get_ood_score_multiprototypes(feature, centroids, emd=False)
        
        # # calculate with faiss
        # index = faiss.IndexFlatL2(centroids.shape[1])
        # index.add(centroids)
        # data_features = get_feature(model_f, data_loader, device)
        # D, _ = index.search(data_features, 1)
        # score = -D[:, -1]
    elif method in ['KNN', 'KNN+']:
        KNN_k = 50
        train_features = get_feature(model_f, train_loader, device)
        index = faiss.IndexFlatL2(train_features.shape[1])
        index.add(train_features)
        data_features = get_feature(model_f, data_loader, device)
        for K in [KNN_k]:
            D, _ = index.search(data_features, K)
            score = -D[:, -1]

    return score