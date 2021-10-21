from collections import Counter as counter
from collections import Counter
import random
import numpy as np

from drift_dataset.concepted_dataset._base import BaseConceptedDataset


class TabulerConceptedDataset(BaseConceptedDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __repr__(self):
        print(f'This concept: {counter(self.targets)}, total: {len(self.targets)}')
        size = 50
        images, y = self.get_random(self.concept_id, size)
        print(f'This split: {counter(y.tolist())}, total: {len(y)}')
        #print(f'mean: {self.data.mean(axis=0)}')
        #print(f'std: {self.data.std(axis=0)}')
        
        return f'current id: {self.concept_id}'
    
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
#         random.Random(self.seed).shuffle(retri_ls)
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
    
    
    
# class TabulerConceptedDataset_normalize(TabulerConceptedDataset):
    
#     def set_concept(self, *args, **kwargs):
#         super().__set_concept__(*args, **kwargs)
#         mean = self.data.mean(axis=0)
#         std = self.data.std(axis=0)
#         self.data = (self.data-mean)/std