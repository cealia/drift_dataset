import numpy as np
import pandas as pd
import glob
import os
import random
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random


class Aging:
    def __init__(self, 
                 concept_ids,
                 seed
                )->None:
        self.concept_ids = concept_ids
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        global used_age_ids, task_num
        dir_name = './download/CAAEData'
        meta = pd.read_csv(os.path.join(dir_name, './CAAEData_g.txt'), sep=",") #CAAEData/01/0_1_7677.jpg  #1_11314
#         (os.path.join(dir_name, f'#{g}', f'#_{pid}.jpg'))
        self.meta = meta
        mapping = {
            0:[0,3,4,1,2],
            1:[4,1,3,2,0],
            2:[1,0,2,4,3],
            3:[2,4,0,3,1],
            4:[3,2,1,0,4]
        }
        real = {
            0:0,
            1:3,
            2:4,
            3:7,
            4:9
        }
        def gid2full(a,gid):
            a = real[a]
            g =  gid.split('_')[0]
            full = [f'{a}{g}', f'{a}_{gid}.jpg']
            return full
        def gid2full_random(a,gid):
            a = random.choice([2*a, 2*a+1])
            g =  gid.split('_')[0]
            full = [f'{a}{g}', f'{a}_{gid}.jpg']
            return full
        data = []
        targets = []
        concepts = []
        for c in concept_ids:
            group_select_class = mapping[c]
            for g,select_cls in enumerate(group_select_class):
                pids = meta[meta.group==g].gid #series
                if g==select_cls:
                    #select_a = real[select_cls]
                    data.extend( pids.apply(lambda x:os.path.join(dir_name,*gid2full(select_cls,x))).values.tolist() )
                else:
                    data.extend(pids.apply\
                                (lambda x:os.path.join(dir_name,*gid2full_random(select_cls,x))).values.tolist())
                targets.extend([select_cls]*len(pids))
                concepts.extend([c]*len(pids))
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.concept_indices = np.array(concepts)
        assert len(self.data)==len(self.targets) and len(self.targets)==len(self.concept_indices), 'different length'
    def __repr__(self): #show concept 0
        mask = self.concept_indices==0
        #print(np.where(mask))
        show = 20
        compose = [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(compose)
        for t in sorted(list(set(self.targets))):
            temp_mask = (self.targets==t) & mask 
            for i in range(show):
                print(f'{self.data[temp_mask][i]}, {self.targets[temp_mask][i]}, \
                {self.concept_indices[temp_mask][i]}')
                pid = self.data[temp_mask][i].split('.jpg')[0].split('/')[-1][2:]
                #print(pid)
                print(self.meta[self.meta.gid==pid].age.values)
                per_fnames = [self.data[temp_mask][i]]
                images = []
                for fname in per_fnames:
                    img = torchvision.io.read_image(fname)
                    img = transform(img)
                    images.append((img+1)/2)
                grid_img = torchvision.utils.make_grid(images, nrow=1)
                plt.figure(figsize=(2,2))
                plt.imshow(grid_img.permute(1, 2, 0))
                plt.show()
        return ''
    def __len__(self):
        return len(self.targets)

    
# # data = Aging([0,1,2], seed=996)
# used_age_ids = [0,3,6,9]#[0, 5, 9]
# task_num = len(used_age_ids)
# class Aging:
#     def __init__(self, 
#                  concept_ids,
#                  seed
#                 )->None:
#         self.concept_ids = concept_ids
#         import numpy as np
#         import random
#         np.random.seed(seed)
#         random.seed(seed)
#         global used_age_ids, task_num
#         dir_name = './download/CAAEData'
#         meta = pd.read_csv(os.path.join(dir_name, './CAAEData.txt'), sep=",") #CAAEData/01/0_1_7677.jpg  #1_11314 
#         units_uids = []
#         #v1 - only one group
#         uni = meta.gid.values
#         unit = []
#         for pid in uni:
#             g = pid.split('_')[0]
#             unit.append(os.path.join(dir_name, f'#{g}', f'#_{pid}.jpg'))
#         units_uids.append(unit)
#         #v1
        
#         group = {_id:np.repeat(np.arange(task_num), len(units_uids[_id])//task_num) \
#                    for _id in range(len(units_uids))}
#         for k in group.keys(): #fill and shuffle group
#             group[k] = np.append(group[k], np.repeat(group[k][-1], len(units_uids[k])-len(group[k])))
#             random.shuffle(group[k])
#         def get_ls(st, l):
#             re = []
#             for i in range(st,l+1):
#                 re.append(i)
#             for i in range(0,st):
#                 re.append(i)
#             return re
#         mapping = {_id:get_ls(_id,(task_num)-1) for _id in range((task_num))}
#         #print(task_num)
#         #print(mapping)
        
#         data = []
#         targets = []
#         concept_indices = []
#         for _id,unit in enumerate(units_uids):
#             temp_data = []
#             temp_targets = []
#             for aid,a in enumerate(used_age_ids):
#                 for uid in unit:
#                     temp_data.append(uid.replace('#', str(a)))
#                 temp_targets.extend([aid]*len(unit))
#             g = group[_id] #retrieve its split group
#             temp_concept = np.array([mapping[tg] for tg in g])
#             temp_concept = np.moveaxis(temp_concept, 0,1)
#             data.append(temp_data)
#             targets.append(temp_targets)
#             concept_indices.append(temp_concept)
#         self.data = np.array(data).flatten()
#         self.targets = np.array(targets).flatten()
#         self.concept_indices = np.array(concept_indices).flatten()
#         assert len(self.data)==len(self.targets) and len(self.targets)==len(self.concept_indices), 'different length'
        
#     def __repr__(self): #show concept 0
#         mask = self.concept_indices==0
#         show = 20
#         compose = [
#             transforms.ToPILImage(),
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#         ]
#         transform = transforms.Compose(compose)
#         for t in sorted(list(set(self.targets))):
#             temp_mask = (self.targets==t) & mask 
#             for i in range(show):
#                 print(f'{self.data[temp_mask][i]}, {self.targets[temp_mask][i]}, \
#                 {self.concept_indices[temp_mask][i]}')
#                 per_fnames = [self.data[temp_mask][i]]
#                 images = []
#                 for fname in per_fnames:
#                     img = torchvision.io.read_image(fname)
#                     img = transform(img)
#                     images.append((img+1)/2)
#                 grid_img = torchvision.utils.make_grid(images, nrow=1)
#                 plt.figure(figsize=(2,2))
#                 plt.imshow(grid_img.permute(1, 2, 0))
#                 plt.show()
#         return ''
#     def __len__(self):
#         return len(self.data)