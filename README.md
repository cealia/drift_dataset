# Drift Dataset

Drift Dataset contain datasets of different drift types — Covariate Drift, Actual Drift, Concept Drift. Also, we provide a PyTorch portal for data loading.
![alt text](https://github.com/cealia/drift_dataset/blob/master/dataset_info.png)

## Download
Download datasets from [link](http://140.112.31.182/download/).

Create a directory /download  with the working directory.

Unzip and move the downloaded directories into /download.

```python
mkdir download
```

## Pytorch Usage
Import
```python
import sys
sys.path.append('YOUR_WORKING_DIRECTORY') 
from drift_dataset.data import Covertype, PoseDataset, AgeX, gas, Review_y, Review_xy
from drift_dataset.concepted_dataset import ImageConceptedDataset, TabulerConceptedDataset
import numpy as np
import random
import os
import pathlib
pro_path = 'YOUR_WORKING_DIRECTORY'
seed = 1234
```
For tabular datasets:
```python
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
'''
initialize data and dataset 
'''
diff = os.path.relpath(pathlib.Path().absolute(), os.path.join(pro_path,'drift_dataset'))
if '.' in diff:
    out_num = 0
else:
    out_num = (diff.split('/').__len__())
run_dir = os.path.join(pathlib.Path().absolute(), '../'*out_num)

valid = 0.1 #ratio for valid set
task_num = 4 #should be the same as the batch no. indicated in the table

data = Review_xy(list(range(task_num)), run_dir)
dataset = TabulerConceptedDataset(data, valid=valid, seed=seed)


'''
set data config
'''
#trainset
dataset.set_concept(t, 'train') #t=0~task_num-1
#validset
dataset.set_concept(t, 'valid') 
#all
dataset.set_concept(t, 'all') 

'''
load data in Pytorch with tqdm progress bar
'''
dataloader = DataLoader(dataset)
pbar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, (x,y) in pbar:
    x = x.to(YOUR_DEVICE)
    y = y.to(YOUR_DEVICE)

pbar.close()
```

For image datasets:
```python
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
'''
initialize data and dataset 
'''
compose = [
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
]
transform = transforms.Compose(compose)
diff = os.path.relpath(pathlib.Path().absolute(), os.path.join(pro_path,'drift_dataset'))
if '.' in diff:
    out_num = 0
else:
    out_num = (diff.split('/').__len__())
run_dir = os.path.join(pathlib.Path().absolute(), '../'*out_num)

valid = 0.1 #ratio for valid set
task_num = 5 #should be the same as the batch no. indicated in the table

data = AgeX(list(range(task_num)), run_dir)
dataset = ImageConceptedDataset(transform, data, valid=valid, seed=seed)


'''
set data config
'''
#trainset
dataset.set_concept(t, 'train') #t=0~task_num-1
#validset
dataset.set_concept(t, 'valid') 
#all
dataset.set_concept(t, 'all') 

'''
load data in Pytorch with tqdm progress bar
'''
dataloader = DataLoader(dataset)
pbar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, (x,y) in pbar:
    x = x.to(YOUR_DEVICE)
    y = y.to(YOUR_DEVICE)

pbar.close()
```
