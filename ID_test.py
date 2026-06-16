import torch
from data_utils import DatasetLoader
from torch.utils.data import DataLoader
import os
# os.environ.pop("CUDA_VISIBLE_DEVICES")
import numpy as np
import pandas as pd
import argparse


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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

def model_test(model, testloader, use_gpu, UQ=False):
    correct, total = 0, 0
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            if UQ:
                evidence = torch.nn.functional.relu(outputs)
                alpha = evidence + 1
                p = alpha / alpha.sum(1, keepdim=True)
                predictions = p.argmax(dim=1, keepdim=True).flatten()
            else:
                predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            acc1, acc2 = accuracy(outputs, labels, topk=(1, 2))
            top1.update(acc1[0], data.size(0))

    # print('top1', top1.avg)
    acc = correct * 100. / total
    err = 100. - acc
    return acc.cpu().numpy(), err.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ID_test")
    parser.add_argument('--model_path', type=str,
                        default=r'./Trained_model_npys') # ./filter_False_lr_pro_0.5_lr_model_0.01_batch_size_32/model/DMPL_HierarchicalClustering_K40_0.5_0.95_last.pt
    parser.add_argument('--save_excel', type=str,
                        default=r'./results/ID_acc.xlsx')
    
    args = parser.parse_args()
    print(vars(args))
    model_path = args.model_path
    model_list = os.listdir(model_path)
    excel_file_path_transposed = args.save_excel  # xlsx for writing results
    dict_results = {}
    for modelname in model_list:
        if ".pt" in modelname:
            print('modelname', modelname)
            model = torch.load(os.path.join(model_path, modelname))
            use_gpu = True
            testset = DatasetLoader('test', r'C:\Users\zhoux\OneDrive\phd thesis\thesis_code\0-dataset\SAR-OOD\MSTAR\SOC') # 
            test_loader = DataLoader(dataset=testset, batch_size=100, shuffle=False)
            acc, err = model_test(model, test_loader, use_gpu, UQ=True)
            print('acc', acc)
            results = dict()
            results['acc'] = acc
            results['err'] = err
            dict_results[modelname] = results

    df_transposed_results = pd.DataFrame(dict_results).T
    with pd.ExcelWriter(excel_file_path_transposed) as writer:
        df_transposed_results.to_excel(writer, header=False, sheet_name='data-test-acc')