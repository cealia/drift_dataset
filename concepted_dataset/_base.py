import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter as counter




class BaseConceptedDataset(Dataset):
    #__metaclass__ = ABCMeta
    def __init__(self, data, valid=0.0, seed=123):
        
        #set seed
        self.seed = seed
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        
        
        #from data
        self.in_data = data.data
        self.in_targets = data.targets
        self.concept_indices = data.concept_indices
        self.concept_ids = data.concept_ids
        
        
        #split train/valid (stratified per concept per class)
        self.valid = valid
        if self.valid: #create train_mask, valid_mask
            if 'pose' in self.in_data[0]: #if isinstance(self.in_data[0], str): 
                #same person appears in either train or valid
                #works because data are stacked in sequence
                sss = StratifiedShuffleSplit(n_splits=1, test_size=valid, random_state=seed)
                sss.get_n_splits(self.in_data[self.concept_indices==self.concept_ids[0]], \
                                 self.in_targets[self.concept_indices==self.concept_ids[0]])
                self.train_mask = np.zeros(self.in_targets[self.concept_indices==self.concept_ids[0]].shape)
                self.valid_mask = np.zeros(self.in_targets[self.concept_indices==self.concept_ids[0]].shape)
                for train_index, valid_index in sss.split(self.in_data[self.concept_indices==self.concept_ids[0]], \
                                 self.in_targets[self.concept_indices==self.concept_ids[0]]):
                    self.train_mask[train_index] = 1
                    self.valid_mask[valid_index] = 1
                self.train_mask = \
                np.expand_dims(self.train_mask, axis=0).repeat(len(self.concept_ids), axis=0).flatten()
                self.valid_mask = \
                np.expand_dims(self.valid_mask, axis=0).repeat(len(self.concept_ids), axis=0).flatten()
            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=valid, random_state=seed)
                m={tup:_id for _id,tup in \
                    enumerate(list(set([(i,j) for i,j in zip(self.in_targets,self.concept_indices)])))} #(class, concept)
                temp_c = [m[(i,j)] for i,j in zip(self.in_targets,self.concept_indices)]
                sss.get_n_splits(self.in_data, temp_c)
                self.train_mask = np.zeros(self.in_targets.shape)
                self.valid_mask = np.zeros(self.in_targets.shape)
                for train_index, valid_index in sss.split(self.in_data, temp_c): #excute only once
                    self.train_mask[train_index] = 1
                    self.valid_mask[valid_index] = 1
            self.train_mask = self.train_mask.astype('bool')
            self.valid_mask = self.valid_mask.astype('bool')
            assert (self.train_mask.sum()+self.valid_mask.sum())==len(self.in_targets), 'train mask and valid mask are not comple'
            
            
        #get index for gem_shuffle
        #min_item_num = len(self.in_data)//len(self.concept_ids)
        #min_item_num = np.unique(self.concept_indices, return_counts=True)[1].min()
        #based on train_mask  #at most drain on the least(k), get from trainset first topk  
#         min_item_num = np.unique(self.concept_indices[self.train_mask], return_counts=True)[1].min()
#         self.gem_shuffle = list(range(min_item_num))
#         random.shuffle(self.gem_shuffle)
        
        
        
    def __getitem__(self,idx):
        self._sanity_check()
        return self.data[idx].astype(np.float32), self.targets[idx]
    
    
    def __len__(self):
        self._sanity_check()
        return len(self.targets)
    
    
    def set_concept(self, i, mode=''):
        self.concept_id = i
        if self.valid:
            if not mode: raise Exception('need to specify train or valid')
                
            if mode=='train':
                self.data = self.in_data[(self.concept_indices==i)&(self.train_mask)]
                self.targets = self.in_targets[(self.concept_indices==i)&(self.train_mask)].tolist()
            elif mode=='valid':
                self.data = self.in_data[(self.concept_indices==i)&(self.valid_mask)]
                self.targets = self.in_targets[(self.concept_indices==i)&(self.valid_mask)].tolist()
            else: #all
                self.data = self.in_data[self.concept_indices==i]
                self.targets = self.in_targets[self.concept_indices==i].tolist()
        else:
            self.data = self.in_data[self.concept_indices==i]
            self.targets = self.in_targets[self.concept_indices==i].tolist()
    
    
    def _sanity_check(self):
        try:
            getattr(self,'data')
        except:
            raise Exception('Please run set_concept before')
            
            
    def retrieve_concept(self): #for joint -> Tensor, Tensor #just get current self.data, self.targets
        self._sanity_check()
        X = self.data
        Y = self.targets
        class temp:
            def __init__(self):
                self.data = torch.from_numpy(X)
                self.targets = torch.from_numpy(np.array(Y))
        return temp()
    
    def get_all(self):
        self._sanity_check()
        X = self.data
        Y = self.targets
        return torch.from_numpy(X), torch.from_numpy(np.array(Y))
    
    
    def get_random(self, concept_id, size): #for gem, joint -> Tensor, Tensor (get from trainset)
        X = self.in_data[(self.concept_indices==concept_id) & (self.train_mask)]
        y = self.in_targets[(self.concept_indices==concept_id) & (self.train_mask)]
        ratio = size/len(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=self.seed, stratify=y)
        #if verbose:
        print(f'distribution: {counter(list(y_test))}, total: {len(X_test)}')
        
        return torch.from_numpy(X_test), torch.from_numpy(y_test)
        
#         assert size<len(self.gem_shuffle), 'Request exceed data size.'
        
#         X = self.in_data[(self.concept_indices==concept_id) & (self.train_mask)][self.gem_shuffle[:size]]
#         Y = self.in_targets[(self.concept_indices==concept_id) & (self.train_mask)][self.gem_shuffle[:size]]
        
#         assert len(np.unique(self.concept_indices[(self.concept_indices==concept_id) & (self.train_mask)][self.gem_shuffle[:size]]))==1
        
#         return torch.from_numpy(X), torch.from_numpy(Y)

    def get_target_values(self)->list:
        return np.unique(self.in_targets)