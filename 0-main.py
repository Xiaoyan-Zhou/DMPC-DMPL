import os
import argparse
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from util import AverageMeter
from loss import Multiprototypes
from data_utils import DatasetLoader
from torch.utils.data import DataLoader, RandomSampler
import random

def setup_seed(seed):
    import random as python_random
    python_random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    torch.manual_seed(seed)         # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)            # Numpy模块的随机种子
    random.seed(seed)               # Python内置随机模块的种子


def main(trainloader, testloader, args, npy_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
    else:
        print("Currently using CPU")

    # load the trained models
    model = torch.load(r'./init_model_npy/resnet18CE.pt')
    print('model load success')


    model = model.cuda()

    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

    cross_entropy = nn.CrossEntropyLoss()
    if args.loss_option == 'DMPL':
        mpl = Multiprototypes(num_classes=args.num_classes, num_centers=args.num_centers, feature_dim=512,
                                         init_way=args.loss_init, centroids_path=args.centroids_path, label_path=args.label_path, uncertainty_th = args.uncertainty_th)
        optimizer2 = torch.optim.SGD(mpl.parameters(), lr=args.lr_pro)
    elif args.loss_option == 'CE':
        args.weight_pro = 0
        mpl = None
        optimizer2 = None

    if args.stepsize > 0:
        scheduler_model = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)
        if mpl != None:
            scheduler_prototype = lr_scheduler.StepLR(optimizer2, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        mpl, model = train(model, cross_entropy, mpl, optimizer_model, optimizer2, trainloader, use_gpu)

        if args.stepsize > 0:
            scheduler_model.step()
            if mpl != None:
                scheduler_prototype.step()

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            acc, err = model_test(model, testloader, use_gpu)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    # #save the centories
    if args.loss_option == 'DMPL':
        np.save(npy_path, mpl.centers.cpu().detach().numpy())

    return model

def train(model, cross_entropy, mpl, optimizer_model, optimizer2, trainloader, use_gpu):

    model.train()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()
    l3_losses = AverageMeter()
    losses = AverageMeter()

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs = model(data)
        # use the model without last fc layer as the feature extractor
        model_f = torch.nn.Sequential(*list(model.children())[:-1])
        model_f = model_f.cuda()
        features = torch.flatten(model_f(data), 1)

        loss1 = cross_entropy(outputs, labels)

        if mpl != None:
            loss2, loss3 = mpl(features, labels)
            loss = loss1 + loss2 * args.weight_pro
            optimizer2.zero_grad()
            if args.filter:
                if loss3 is not None:
                    loss3.backward(retain_graph=True)
            else:
                loss2.backward(retain_graph=True)
            optimizer2.step()  # 更新mpl对应的参数
        else:
            loss = loss1

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        losses.update(loss.item(), labels.size(0))
        l1_losses.update(loss1.item(), labels.size(0))
        if mpl != None:
            l2_losses.update(loss2.item(), labels.size(0))
            if loss3 is not None:
                l3_losses.update(loss3.item(), labels.size(0))


        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) CrossEntropy Loss {:.6f} ({:.6f}) MPL loss {:.6f} ({:.6f}) MPL_filter {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, l1_losses.val, l1_losses.avg,
                          l2_losses.val, l2_losses.avg, l3_losses.val, l3_losses.avg))

    return mpl, model


def model_test(model, testloader, use_gpu):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DMPL")
    # optimization
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_model', type=float, default=0.01, help="learning rate for model")
    parser.add_argument('--lr_pro', type=float, default=0.5, help="learning rate for prototype loss")
    parser.add_argument('--weight_pro', type=float, default=0.5, help="weight for prototype loss")
    parser.add_argument('--weight_list',  nargs='+', type=int, default=[5], help="weightlist for center loss")
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--stepsize', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
    # model and loss
    parser.add_argument('--filter', action='store_true', help='able the uncertainty filter')# use uncertainty filter in prototype updating
    parser.add_argument('--loss_init', type=str, default='HierarchicalClustering', choices=['normal', 'uniform', 'HierarchicalClustering'])
    parser.add_argument('--loss_option', type=str, default='DMPL', choices=['DMPL', 'CE'])
    parser.add_argument('--num_centers', type=int, default=3, help='it only works when chosing normal and uniformal as loss init')
    parser.add_argument('--K', type=int, default=40)
    parser.add_argument('--centroids_path', type=str, default='./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_centroids.npy', help='The path of centroids obtained by HierarchicalClustering.'
                                                                         ' centroids_path is essential if the loss_init is HierarchicalClustering')
    parser.add_argument('--label_path', type=str, default='./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_labels.npy', help='The path of centroids label obtained by HierarchicalClustering.'
                                                                     'label_path is essential if the loss_init is HierarchicalClustering')

    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use-cpu', action='store_true')
    parser.add_argument('--save-dir', type=str, default='log')
    
    parser.add_argument('--uncertainty_th', type=float, default=0.90)

    parser.add_argument('--data_path', type=str,
                        default=r'C:\Users\zhoux\OneDrive\phd thesis\thesis_code\0-dataset\SAR-OOD\MSTAR\SOC')
    parser.add_argument('--num_classes', type=int, default=10)
    
    ##save
    parser.add_argument('--root', type=str,
                        default=r'./')
    
    args = parser.parse_args()
    
    sub_root_name = 'filter_{}_lr_pro_{}_lr_model_{}_batch_size_{}'.format(str(args.filter), str(args.lr_pro), str(args.lr_model), str(args.batch_size))
    
    model_save_root = os.path.join(args.root, sub_root_name+'/model/')
    npy_save_root = os.path.join(args.root, sub_root_name+'/npy/')

    if not os.path.exists(model_save_root):
        os.makedirs(model_save_root)
        
    if not os.path.exists(npy_save_root):
        os.makedirs(npy_save_root)

    for weight in args.weight_list:
        setup_seed(1)
        args.weight_pro = round(weight*0.1, 1)
        print(vars(args))

        trainset = DatasetLoader('train', args.data_path)
        generator = torch.Generator().manual_seed(42)
        trainsampler = RandomSampler(trainset, generator=generator)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, sampler=trainsampler)

        testset = DatasetLoader('test', args.data_path)
        test_loader = DataLoader(dataset=testset, batch_size=8, shuffle=False, num_workers=0)
    
        name = args.loss_option +'_' + args.loss_init +'_K'+ str(args.K)+'_'+str(args.weight_pro)+'_'+str(args.uncertainty_th)
        npy_path = os.path.join(npy_save_root, name+'_last.npy')
        
        last_model = main(train_loader, test_loader, args, npy_path)

        torch.save(last_model, os.path.join(model_save_root, name+'_last.pt'))
       



