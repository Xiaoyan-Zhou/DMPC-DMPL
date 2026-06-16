import numpy as np
import matplotlib.pyplot as plt
import torch
from ID_test import *
import torch.nn.functional as F

import numpy as np

def aurc_ood(id_score, ood_score, higher_score_is_id=True):
    id_score = np.asarray(id_score).reshape(-1)
    ood_score = np.asarray(ood_score).reshape(-1)

    scores = np.concatenate([id_score, ood_score], axis=0)
    labels = np.concatenate([np.ones_like(id_score), np.zeros_like(ood_score)], axis=0)  # 1=ID,0=OOD

    if not higher_score_is_id:
        scores = -scores

    idx = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[idx]

    n = labels_sorted.shape[0]
    k = np.arange(1, n + 1, dtype=np.float64)
    cum_id = np.cumsum(labels_sorted)  # accepted中ID个数
    risk = 1.0 - (cum_id / k)          # = accepted中OOD占比
    coverage = k / n

    aurc = np.trapz(risk, coverage)
    return aurc, coverage, risk

# calculate the AURC of ID
def risk_coverage_from_msp(probs: np.ndarray, labels: np.ndarray):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n = probs.shape[0]

    conf = probs.max(axis=1)          # MSP confidence
    pred = probs.argmax(axis=1)
    err  = (pred != labels).astype(np.float64)  # 0/1 error

    # sort by confidence descending
    idx = np.argsort(-conf, kind="mergesort")
    err_sorted = err[idx]

    # cumulative risk for top-k accepted samples
    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cum_err / k

    # AURC = area under risk-coverage curve
    aurc = np.trapz(risk, coverage)
    return aurc, coverage, risk

def plot_aurc_curve(probs, labels, title="Risk–Coverage (AURC) Curve"):
    aurc, coverage, risk = risk_coverage_from_msp(probs, labels)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(coverage, risk)
    plt.xlabel("Coverage (fraction accepted)")
    plt.ylabel("Risk (error rate on accepted)")
    plt.title(f"{title}\nAURC = {aurc:.6f}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return aurc

def get_probs_and_labels(model, testloader, use_gpu=True):
    """
    返回:
      probs: (n, C) numpy float
      labels: (n,) numpy int
      acc: float (%)
      err: float (%)
    """
    model.eval()

    all_probs = []
    all_labels = []
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data = data.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            outputs = model(data)
            probs = F.softmax(outputs, dim=1)               # (B, C)
            preds = probs.argmax(dim=1)                         # (B,)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    acc = 100.0 * correct / total
    err = 100.0 - acc
    return probs, labels, acc, err

if __name__ == '__main__':

    parser = argparse.ArgumentParser("ID test AURC")
    parser.add_argument('--model_path', type=str,
                        default=r'./filter_True_lr_pro_0.5_lr_model_0.01_batch_size_32/model/DMPL_HierarchicalClustering_K40_0.5_0.9_last.pt') 
    parser.add_argument('--save_excel', type=str,
                        default=r'./TGRS_revision/ID_acc.xlsx') 
    args = parser.parse_args()
    print(vars(args))
    
    model_path = args.model_path
    excel_file_path_transposed = args.save_excel  # xlsx for writing results
    model = torch.load(model_path)
    use_gpu = True
    testset = DatasetLoader('test', r'C:\Users\zhoux\OneDrive\phd thesis\thesis_code\0-dataset\SAR-OOD\MSTAR\SOC') # 
    test_loader = DataLoader(dataset=testset, batch_size=100, shuffle=False)
    probs, labels, acc, err = get_probs_and_labels(model, test_loader, use_gpu=True)
    print('acc', acc)

    # probs: softmax output (n, C)
    # labels: ground truth (n,)
    aurc_value = plot_aurc_curve(probs, labels)
    print("AURC:", aurc_value)