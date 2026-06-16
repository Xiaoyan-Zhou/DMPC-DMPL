# Clustering-based Prototype Generation Stage 
# Contrastive method
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib
import matplotlib.pyplot as plt

from data_utils import DatasetLoader

def get_device(device_number):
    use_cuda = torch.cuda.is_available()
    if use_cuda and device_number in ['0', '1']:
        device = torch.device(f"cuda:{device_number}")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

if __name__ == '__main__':
    num_classes = 10  # Assume ten classes
    prototypes_per_class = 1  # Assume 3 prototypes per class

    # Configuration for paths based on the granularity of the dataset
    data_path = r'C:\Users\zhoux\OneDrive\phd thesis\thesis_code\0-dataset\SAR-OOD\MSTAR\SOC'
    model_path = './init_model_npy/resnet18CE.pt'
    centroids_path = f'./init_model_npy/fixed/ward/resnet18CE_10_K{prototypes_per_class*num_classes}_HierarchicalClustering_centroids_fixed.npy'
    labels_path = f'./init_model_npy/fixed/ward/resnet18CE_10_K{prototypes_per_class*num_classes}_HierarchicalClustering_labels_fixed.npy'

    # Load the trained model and dataset
    model = torch.load(model_path, map_location='cpu')
    model_f = torch.nn.Sequential(*list(model.children())[:-1])
    device = get_device('0')
    model_f = model_f.to(device)

    trainset = DatasetLoader('train', data_path)
    train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False)

    model.eval()
    features_list = []
    label_true = []
    with torch.no_grad():
        for images, label in tqdm(train_loader, total=len(train_loader)):
            images = images.to(device)
            features = model_f(images)
            features_list.append(features.cpu().numpy().flatten())
            label_true.append(label.item())

    features_array = np.array(features_list)
    label_array = np.array(label_true)

    # Cluster per class
    centroids = []
    labels = []
    for i in range(num_classes):
        class_features = features_array[label_array == i]
        if len(class_features) < prototypes_per_class:
            print(f"Not enough samples for class {i} for clustering.")
            continue
        Z = linkage(class_features, method='ward')#'single'
        clusters = fcluster(Z, t=prototypes_per_class, criterion='maxclust')
        class_centroids = np.array([class_features[clusters == k].mean(axis=0) for k in range(1, prototypes_per_class + 1)])
        centroids.append(class_centroids)
        labels.append(np.full((prototypes_per_class,), i))

    centroids = np.vstack(centroids)
    labels = np.hstack(labels)

    # Save the centroids and labels
    np.save(centroids_path, centroids)
    np.save(labels_path, labels)
