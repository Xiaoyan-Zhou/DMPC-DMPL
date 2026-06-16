import torch
import torch.nn as nn
import numpy as np

class Multiprototypes(nn.Module):
    def __init__(self, num_classes=10, num_centers=3, feature_dim=512, use_gpu=True, init_way='normal', centroids_path=None, label_path=None, uncertainty_th =0.9):
        super(Multiprototypes, self).__init__()
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.use_gpu = use_gpu
        self.init_way = init_way
        self.th = uncertainty_th

        if init_way == 'normal':
            # Initialize the class center parameters. A simple normal distribution initialization is used here, with a mean of 0 and a variance of 1.
            if self.use_gpu:
                self.centers = nn.Parameter(torch.randn(num_classes, num_centers, feature_dim).cuda())
            else:
                self.centers = nn.Parameter(torch.randn(num_classes, num_centers, feature_dim))
        elif init_way == 'uniform':
            # uniform distribution
            min_val = 0
            max_val = 30 #n=10
            if self.use_gpu:
                self.centers = nn.Parameter(
                    torch.FloatTensor(num_classes, num_centers, feature_dim).uniform_(min_val, max_val).cuda())
            else:
                self.centers = nn.Parameter(
                    torch.FloatTensor(num_classes, num_centers, feature_dim).uniform_(min_val, max_val))
        elif init_way in ['HierarchicalClustering']:
            if centroids_path != None and label_path != None:
                center_array = np.load(centroids_path)
                center_label = np.load(label_path)
            else:
                print('Error: There is no centroids path or label path')
            if self.use_gpu:
                self.centers = nn.Parameter(torch.FloatTensor(center_array).cuda())
                self.center_label = torch.tensor(center_label).cuda()
            else:
                self.centers = nn.Parameter(torch.FloatTensor(center_array))
                self.center_label = torch.tensor(center_label)

    def get_entropy(self, distance):
        eps = 1e-12
        distance = distance+eps
        p = distance[0]/(distance[0]+distance[1])
        return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))

    def forward(self, x, labels):
        loss1 = None
        loss2 = None
        batch_size = x.size(0)
        if self.init_way == 'HierarchicalClustering':
            mask = torch.full((x.shape[0],), True, dtype=torch.bool)
            # Calculate the distance of all samples to all prototypes
            distances = (x.unsqueeze(1) - self.centers.unsqueeze(0)).pow(2).sum(dim=-1) # [batch_size, num_prototypes]
            min_distances = torch.full((x.shape[0],), float('inf'), device=x.device)
            # For each sample, find the smallest distance to the prototype of the same class
            for i, label in enumerate(labels):
                # Get the prototype index corresponding to this category
                same_class_indices = [idx for idx, p_label in enumerate(self.center_label) if p_label == label]
                # If there is a similar prototype, update the minimum distance
                if same_class_indices:  # same_class_indices
                    min_distances[i] = distances[i, same_class_indices].min()
                    sorted_values, sorted_indices = torch.sort(distances[i, same_class_indices])
                    if (len(same_class_indices) > 1) and (self.get_entropy(sorted_values[:2]) > self.th):
                        mask[i] = False
            # calculate loss
            loss1 = min_distances.mean()
            loss2 = min_distances[mask].mean()
        else:
            #distances = (x.unsqueeze(1).unsqueeze(2) - self.centers.unsqueeze(0)).pow(2).sum(dim=-1)
            #labels_expand = labels.view(-1, 1, 1).expand(batch_size, 1, self.num_centers)
            #positive_distances = distances.gather(1, labels_expand).squeeze(1)
            #min_positive_distances, _ = positive_distances.min(dim=1, keepdim=True)
            #loss1 = min_positive_distances.mean()
            
            mask = torch.full((x.shape[0],), True, dtype=torch.bool)
            distances = (x.unsqueeze(1).unsqueeze(2) - self.centers.unsqueeze(0)).pow(2).sum(dim=-1)
            labels_expand = labels.view(-1, 1, 1).expand(batch_size, 1, self.num_centers)
            positive_distances = distances.gather(1, labels_expand).squeeze(1)
            min_positive_distances, _ = positive_distances.min(dim=1, keepdim=True)
            for i in range(positive_distances.shape[0]):
                sorted_values, sorted_indices = torch.sort(positive_distances[i])
                if (self.num_centers > 1) and (self.get_entropy(sorted_values[:2]) > 0.95):
                    mask[i] = False
            loss1 = min_positive_distances.mean()
            loss2 = min_positive_distances[mask].mean()
        return loss1, loss2
