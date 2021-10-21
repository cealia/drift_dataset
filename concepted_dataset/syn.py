import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
# import random
# seed = 123
# np.random.seed(seed)
# random.seed(seed)

from drift_dataset.concepted_dataset._base import BaseConceptedDataset

class synConceptedDataset(BaseConceptedDataset):
    
    def __repr__(self):
        self._sanity_check()
        plt.scatter(self.data[:,0], self.data[:,1], alpha=0.1, s=10, c=self.targets)
        plt.show()
        return f'current id: {self.concept_id}'
    
    def plot_boundary(self, pred_fn, show:list):
        self.xmax = np.max(self.in_data[:,0])
        self.xmin = np.min(self.in_data[:,0])
        self.ymax = np.max(self.in_data[:,1])
        self.ymin = np.min(self.in_data[:,1])
        h = 0.02
        a = np.arange(self.xmin, self.xmax, h)
        b = np.arange(self.ymin, self.ymax, h)
        xx, yy = np.meshgrid(a, b)
        x_plot = []
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x_plot.append([xx[i][j], yy[i][j]])
        x_plot = torch.FloatTensor(x_plot)
        #x_plot_dataset = TensorDataset(x_plot)
        class SimpleDataset(Dataset):
            def __init__(self,data):
                self.data = data
            def __getitem__(self,idx):
                return self.data[idx]
            def __len__(self):
                return len(self.data)
        x_plot_dataset = SimpleDataset(x_plot)
        dataloader = DataLoader(x_plot_dataset, batch_size=4096,shuffle=False)
        
        r = pred_fn(dataloader)
        preds = np.array(r)
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.xlabel('x1')
        plt.ylabel('x2')
        colors = ['#FF0000', '#00FF00']
        for pred,color in zip(preds,colors):
            zz = np.array(pred).reshape(xx.shape)
            plt.contour(xx, yy, zz, [0.5], colors=color, linewidths=1)
        zz = np.sum(preds,axis=0).reshape(xx.shape)
        plt.contourf(xx, yy, zz, [0.5,1.5], colors=['#E3B7C5', '#E1E688', '#B9B7E3'], extend='both')
        ax.set_xlim([np.amin(xx), np.amax(xx)])
        ax.set_ylim([np.amin(yy), np.amax(yy)])
        ax.set_xticks(np.arange(int(np.amin(xx)), int(np.amax(xx))+1, 1))
        ax.set_yticks(np.arange(int(np.amin(yy)), int(np.amax(yy))+1, 1))
        mask = np.isin(self.concept_indices, show)
        ax.scatter(self.in_data[mask,0], self.in_data[mask, 1], s=1, c=self.in_targets[mask], marker='o')
        plt.show()