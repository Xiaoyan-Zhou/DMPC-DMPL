from ood_utils import get_device, get_ood_score, plot_distribution
import os
import torch
from data_utils import DatasetLoader
from torch.utils.data import DataLoader
import metrics
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from AURC import aurc_ood

def plot_selective_risk_curve(id_score, ood_score, title, savepath, higher_score_is_id=True):
    aurc, coverage, risk = aurc_ood(
        id_score=id_score,
        ood_score=ood_score,
        higher_score_is_id=higher_score_is_id
    )

    id_score = np.asarray(id_score).reshape(-1)
    ood_score = np.asarray(ood_score).reshape(-1)
    random_risk = len(ood_score) / max(len(id_score) + len(ood_score), 1)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(coverage, risk, linewidth=2.0, label=f"Curve (AURC={aurc:.6f})")
    plt.axhline(y=random_risk, color='gray', linestyle='--', linewidth=1.2, label='Random baseline')
    plt.xlabel("Coverage")
    plt.ylabel("Selective Risk")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()

    return aurc

if __name__ == '__main__':
    dict_results_sample = {}
    dict_results_SAR_ACD = {}
    dict_results_FUSAR_ship = {}
    dict_results_MSTAR_OOD = {}

    test_model = False # Whether to batch test models or not

    parser = argparse.ArgumentParser("OOD_test")
    parser.add_argument('--model_root', type=str,
                        default=r'./Trained_model_npys')
    parser.add_argument('--npy_root', type=str,
                        default=r'./Trained_model_npys')
    parser.add_argument('--save_excel', type=str,
                        default=r'./results/OOD_results_with_AURC.xlsx')
    parser.add_argument('--fig_results', type=str,
                        default=r'./results/')
    
    args = parser.parse_args()
    os.makedirs(args.fig_results, exist_ok=True)
    selective_curve_dir = os.path.join(args.fig_results, 'selective_risk_curve')
    os.makedirs(selective_curve_dir, exist_ok=True)

    DMPC_model_path = r'./init_model_npy/resnet18CE.pt' # class number: 10
    DMPC_centroids_path = r'./init_model_npy/resnet18_CE_10_K70_HierarchicalClustering_centroids.npy'  # obtained by clustering

    DMPL_centroids_path = r'./Trained_model_npys/DMPL_HierarchicalClustering_K40_0.5_0.9_last.npy'# obtained by DMPL
    DMPL_model_path = r'./Trained_model_npys/DMPL_HierarchicalClustering_K40_0.5_0.9_last.pt' #trained model supervised by JPEL
    
    method_list = ['MaxNorm', 'MSP', 'ODIN', 'Energy', 'GradNorm', 'MaxLogit', 'KNN',  'DML', 'DMPC', 'DMPL']
    # method_list = ['DMPL']

    for method in method_list:

        name = method
        if method == 'DMPL':
            try:
                model = torch.load(DMPL_model_path)
                print('model path', DMPL_model_path)
            except:
                print('can not load model')
        else:
            model = torch.load(DMPC_model_path)

        # The path for the trained model
        # The path for the centroids file
        centroids = None
        train_loader = None
        normalize_value = None

        device = get_device('0')

        model_f = torch.nn.Sequential(*list(model.children())[:-1])
        model_f = model_f.to(device)
        model = model.to(device)
        model_f = model_f.to(device)

        data_test_path = r"C:\Users\zhoux\OneDrive\phd thesis\thesis_code\dataset_paper\1-OODdata\MSTAR\SOC"
        dataset = DatasetLoader('test', data_test_path)
        id_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        #ood test SAR SAMPLE
        data_test_path = r"C:\Users\zhoux\OneDrive\phd thesis\thesis_code\dataset_paper\1-OODdata\SAMPLE"
        dataset_sample = DatasetLoader('ood', data_test_path)
        ood_sample_loader = DataLoader(dataset=dataset_sample, batch_size=1, shuffle=False)

        # ood test FUSAR-ship
        data_test_path = r'C:\Users\zhoux\OneDrive\phd thesis\thesis_code\dataset_paper\1-OODdata\SHIP\FUSAR-ship'
        dataset_ship = DatasetLoader('ood', data_test_path)
        ood_ship_loader = DataLoader(dataset=dataset_ship, batch_size=1, shuffle=False)

        # ood test SAR-ACD
        data_path = r'C:\Users\zhoux\OneDrive\phd thesis\thesis_code\dataset_paper\1-OODdata\AIRPLANE\SAR-ACD-main'
        ood_testset_airplane = DatasetLoader('ood', data_path)
        ood_airplane_loader = DataLoader(dataset=ood_testset_airplane, batch_size=1,
                                         shuffle=False)  # set batch_size=1 when test uncertainty

        # ###################### multiprototypes based ood detection ##########################
        #load centroids of ID data
        if method == 'DMPL':
            print('##############################')
            print('DMPL_centroids_path', DMPL_centroids_path)
            centroids = np.load(DMPL_centroids_path)
            print('centroids.shape', centroids.shape)
            if len(centroids.shape) == 3:
                centroids = centroids.reshape(-1, centroids.shape[-1])
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / norms
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader)
        elif method == 'DMPC':
            print('##############################')
            print('DMPC_centroids_path', DMPC_centroids_path)
            centroids = np.load(DMPC_centroids_path)
            print('centroids.shape', centroids.shape)
            if len(centroids.shape) == 3:
                centroids = centroids.reshape(-1, centroids.shape[-1])
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / norms
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader)
        elif method in ['KNN', 'KNN+']:
            #################### KNN score OOD detection #################
            # data_train_path = r'/scratch/project_2002243/zhouxiaoyan/SAR-OOD/MSTAR/SOC'
            data_train_path = r'C:\Users\zhoux\OneDrive\phd thesis\thesis_code\dataset_paper\1-OODdata\MSTAR\SOC'
            trainset = DatasetLoader('train', data_train_path)
            train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False)
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader, normalize_value=None)
        elif method in ['MaxLogit', 'DML', 'MaxNorm', 'MSP', 'Energy']:
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader, normalize_value=None)
            normalize_value = np.sum(id_score)
            id_score = id_score / normalize_value
        else:
            id_score = get_ood_score(id_loader, model, model_f, device, method, centroids, train_loader, normalize_value=None)

        #calculate score with method
        sample_score = get_ood_score(ood_sample_loader, model, model_f, device, method, centroids, train_loader, normalize_value)
        FUSAR_ship_score = get_ood_score(ood_ship_loader, model, model_f, device, method, centroids, train_loader, normalize_value)
        SAR_ACD_score = get_ood_score(ood_airplane_loader, model, model_f, device, method, centroids, train_loader, normalize_value)

        results_sample = metrics.cal_metric(id_score, sample_score)
        aurc_sample, _, _ = aurc_ood(id_score, sample_score, higher_score_is_id=True) ## revision for TGRS
        results_sample["AURC"] = aurc_sample ## revision for TGRS
        plot_selective_risk_curve(
            id_score=id_score,
            ood_score=sample_score,
            title=f"{name}: MSTAR vs SAMPLE",
            savepath=os.path.join(selective_curve_dir, f"{name}_sample_selective_risk.png"),
            higher_score_is_id=True
        )
        print(name+': sample', results_sample)
        dict_results_sample[name] = results_sample

        results_airplane = metrics.cal_metric(id_score, SAR_ACD_score)
        aurc_airplane, _, _ = aurc_ood(id_score, SAR_ACD_score, higher_score_is_id=True) ## revision for TGRS
        results_airplane["AURC"] = aurc_airplane## revision for TGRS
        plot_selective_risk_curve(
            id_score=id_score,
            ood_score=SAR_ACD_score,
            title=f"{name}: MSTAR vs SAR-ACD",
            savepath=os.path.join(selective_curve_dir, f"{name}_sar_acd_selective_risk.png"),
            higher_score_is_id=True
        )
        print(name+': airplane', results_airplane)
        dict_results_SAR_ACD[name] = results_airplane

        results_ship = metrics.cal_metric(id_score, FUSAR_ship_score)
        aurc_ship, _, _ = aurc_ood(id_score, FUSAR_ship_score, higher_score_is_id=True) ## revision for TGRS
        results_ship["AURC"] = aurc_ship## revision for TGRS
        plot_selective_risk_curve(
            id_score=id_score,
            ood_score=FUSAR_ship_score,
            title=f"{name}: MSTAR vs FUSAR-ship",
            savepath=os.path.join(selective_curve_dir, f"{name}_fusar_ship_selective_risk.png"),
            higher_score_is_id=True
        )
        print(name+': ship', results_ship)
        dict_results_FUSAR_ship[name] = results_ship

        plot_distribution([id_score, sample_score, SAR_ACD_score, FUSAR_ship_score],
                 ['MSTAR (ID)', 'SAMPLE (OOD)', 'SAR-ACD (OOD)', 'FUSAR-ship (OOD)'],
                 savepath=os.path.join(args.fig_results, name + '.png'), alpha=0.5) #dawn

    # Convert the dictionary to a DataFrame and then transpose it
    df_transposed_sample = pd.DataFrame(dict_results_sample).T
    df_transposed_SAR_ACD = pd.DataFrame(dict_results_SAR_ACD).T
    df_transposed_FUSAR_ship = pd.DataFrame(dict_results_FUSAR_ship).T

    with pd.ExcelWriter(args.save_excel) as writer:
        df_transposed_sample.to_excel(writer, header=False, sheet_name='sample')
        df_transposed_SAR_ACD.to_excel(writer, header=False, sheet_name='SAR_ACD')
        df_transposed_FUSAR_ship.to_excel(writer, header=False, sheet_name='FUSAR_ship')




