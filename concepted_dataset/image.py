import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import torchvision
import pickle
import matplotlib.pyplot as plt
import random
from collections import Counter

from drift_dataset.concepted_dataset._base import BaseConceptedDataset



# add transform for image
class ImageConceptedDataset(BaseConceptedDataset):
    
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        
        
    def __getitem__(self,idx):
        self._sanity_check()
        fname = self.data[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img, self.targets[idx]
    
    
    def retrieve_concept(self): #for joint -> Tensor, Tensor
        self._sanity_check()

        X = self.data
        Y = self.targets
        X_img = []
        for fname in X:
            img = torchvision.io.read_image(fname)
            img = self.transform(img)
            X_img.append(img)
        class temp:
            def __init__(self):
                self.data = torch.stack(X_img)
                self.targets = torch.from_numpy(np.array(Y))
        return temp()
    
    def get_all(self):
        self._sanity_check()
        X = self.data
        Y = self.targets
        X_img = []
        for fname in X:
            img = torchvision.io.read_image(fname)
            img = self.transform(img)
            X_img.append(img)
        return torch.stack(X_img), torch.from_numpy(np.array(Y))
    
    
    def get_random(self, concept_id, size): #for gem -> Tensor, Tensor
        X = self.in_data[(self.concept_indices==concept_id) & (self.train_mask)]
        y = self.in_targets[(self.concept_indices==concept_id) & (self.train_mask)]
        ratio = size/len(X)
        if 'pose' in self.in_data[0]:
            X_train, X, y_train, Y = train_test_split(X, y, test_size=ratio, random_state=self.seed+concept_id, stratify=y)
        else:
            X_train, X, y_train, Y = train_test_split(X, y, test_size=ratio, random_state=self.seed, stratify=y)
        
        
#         assert size<len(self.gem_shuffle), 'Request exceed data size.'
# #         X = self.in_data[self.concept_indices==concept_id][self.gem_shuffle[:size]]
# #         Y = self.in_targets[self.concept_indices==concept_id][self.gem_shuffle[:size]]
# #         assert len(np.unique(self.concept_indices[self.concept_indices==concept_id][self.gem_shuffle[:size]]))==1

#         X = self.in_data[(self.concept_indices==concept_id) & (self.train_mask)][self.gem_shuffle[:size]]
#         Y = self.in_targets[(self.concept_indices==concept_id) & (self.train_mask)][self.gem_shuffle[:size]]
        
#         assert len(np.unique(self.concept_indices[(self.concept_indices==concept_id) & (self.train_mask)][self.gem_shuffle[:size]]))==1
        
    
        X_img = []
        for fname in X:
            img = torchvision.io.read_image(fname)
            img = self.transform(img)
            X_img.append(img)
            
        return torch.stack(X_img), torch.from_numpy(Y)
    
    
    def __repr__(self):
        from collections import Counter as counter
        size = 50
        images, y = self.get_random(self.concept_id, size)
        
        #render concept images
        grid_img = torchvision.utils.make_grid((images+1)/2, nrow=10)
        plt.figure(figsize=(100,100))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        
        #show statistic
        print(f'This concept: {counter(self.targets)}, total: {len(self.targets)}')
        print(f'This split: {counter(y.tolist())}, total: {len(y)}')
        
        return f'current id: {self.concept_id}'
        
        
#         size = 50
#         assert size<len(self.gem_shuffle), 'Request exceed data size.'
#         self._sanity_check()
        
#         #use gem random shuffle to display
#         concept_id = self.concept_id
#         cs = self.in_data[(self.concept_indices==concept_id) & (self.train_mask)][self.gem_shuffle[:size]]
#         #self.data[self.gem_shuffle[:size]]
        
#         #render concept images
#         images = []
#         for fname in cs:
#             img = torchvision.io.read_image(fname)
#             img = self.transform(img)
#             images.append((img+1)/2) #de-normalize
#         grid_img = torchvision.utils.make_grid(images, nrow=10)
#         plt.figure(figsize=(100,100))
#         plt.imshow(grid_img.permute(1, 2, 0))
#         plt.show()
        
#         #show statistic
# #         print(counter(self.in_targets[self.concept_indices==self.concept_id][self.gem_shuffle[:size]]))
#         print(f'This concept: {counter(self.targets)}, total: {len(self.targets)}')
#         #print(self.gem_shuffle[:size])
#         print(f'This split: {counter(self.in_targets[(self.concept_indices==concept_id) & (self.train_mask)][self.gem_shuffle[:size]])}, total: {len(self.in_data[(self.concept_indices==concept_id) & (self.train_mask)])}')
        
#         return f'current id: {self.concept_id}'

#     def get_mixture(self):
#         data, targets = [],[]
#         for i in self.concept_ids[:-1]:
#             self.set_concept(i, 'valid')
#             print(self.concept_id)
#             data.append(self.data)
#             targets.append(self.targets)
#         old_num = len(data)
#         self.set_concept(self.concept_ids[-1], 'all')
#         print(self.concept_id)
#         new_num = len(self.data)
#         retri_ls = list(range(new_num))
#         random.shuffle(retri_ls)
#         data.append(self.data[retri_ls[:old_num]])
#         np_t = np.array(self.targets)
#         targets.append(np_t[retri_ls[:old_num]])
#         data = np.concatenate(data, axis=0)
#         targets = np.concatenate(targets, axis=0)
#         print(data.shape, targets.shape)
#         print(Counter(targets))
#         self.data = data
#         self.targets = targets
#         return #data, targets.tolist()

    def get_mixture(self):
        data, targets = [],[]
        for i in self.concept_ids:
            self.set_concept(i, 'valid')
            print(self.concept_id)
            data.append(self.data)
            targets.append(self.targets)
#         old_num = len(data)
#         self.set_concept(self.concept_ids[-1], 'all')
#         print(self.concept_id)
#         new_num = len(self.data)
#         retri_ls = list(range(new_num))
#         random.shuffle(retri_ls)
#         data.append(self.data[retri_ls[:old_num]])
#         np_t = np.array(self.targets)
#         targets.append(np_t[retri_ls[:old_num]])
        data = np.concatenate(data, axis=0)
        targets = np.concatenate(targets, axis=0)
        print(data.shape, targets.shape)
        print(Counter(targets))
        self.data = data
        self.targets = targets
        return #data, targets.tolist()