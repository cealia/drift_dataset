# task_num = 10
# valid = 0.2
# seed = 1234
# from datasets.syn1 import circle
# dconfig = {
#     'r':3,
#     'concept_num':task_num,
#     'cov':[[0]*i+[0.18]+[0]*(2-i-1) for i in range(2)],
#     'num_per_concept': 2000,
#     'seed':seed
# }
# data = circle(dconfig)
# print(data)

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import matplotlib
from torch.utils.data import Dataset
import torch
import math

# config = {
#     'r':3,
#     'concept_num':5,
#     'cov':[[0]*i+[0.1]+[0]*(2-i-1) for i in range(2)],
#     'num_per_concept': 1000,
# }

class circle:
    def __init__(self,config):
        self.config = config
        self.gen_data()
    def gen_data(self):
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        r = self.config['r']
        xa = 0
        ya = -self.config['r']
        x0 = 0
        y0 = 0
        alfa = 360/self.config['concept_num']
        list_x=[]
        list_y=[]
        N=self.config['concept_num']
        for i in range(N):
            theta=math.atan2(ya,xa)
            theta2=math.atan2(ya,xa)+math.pi
            x=x0+r*math.cos(theta-alfa*math.pi/180)
            y = y0 + r * math.sin(theta - alfa*math.pi/180)
            list_x.append(x)
            list_y.append(y)
            xa=x
            ya=y
        self.data = []
        self.targets = []
        self.concept_indices = []
        self.concept_ids = list(range(self.config['concept_num']))
        for c in range(self.config['concept_num']):
            p = np.random.multivariate_normal([list_x[c],list_y[c]], self.config['cov'], self.config['num_per_concept'])
            self.data.extend(p)
            self.targets.extend([0]*self.config['num_per_concept'])
            p = np.random.multivariate_normal([0,0], self.config['cov'], self.config['num_per_concept'])
            self.data.extend(p)
            self.targets.extend([1]*self.config['num_per_concept'])
            self.concept_indices.extend([c]*(2*self.config['num_per_concept']))
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.concept_indices = np.array(self.concept_indices)
        assert len(self.data)==len(self.targets)
        assert len(self.targets)==len(self.concept_indices)
    def __repr__(self):
        sha = {s:i for i,s in enumerate(list(set([(cid,t) for cid,t in zip(self.concept_indices,self.targets)])))}
        label = [sha[s] for s in [(cid,t) for cid,t in zip(self.concept_indices,self.targets)]]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(self.data[:,0], self.data[:,1], alpha=0.1, s=10, c=label, marker='o')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
        return ''
    def __len__(self):
        return len(self.data)