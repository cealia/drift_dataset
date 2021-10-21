import pandas as pd
import numpy as np
import os
import pickle

# compose = [
#     transforms.ToPILImage(),
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
# ]
# transform = transforms.Compose(compose)    
# data = PoseDataset(list(range(5)))
# dataset = ImageConceptedDataset(transform, data, valid=0.1)
# sp = ['train','valid']
# for i in range(5):
#     for s in sp:
#         # len(data)
#         dataset.set_concept(i, s)
#         # len(dataset)
#         print(dataset)

people = 10000
selected_pose = [1,6,9,13,19]
thre = 0.95
#add criteria = 'gender', 'dominant_race', 'dominant_emotion'
class PoseDataset:
    def __init__(self, 
                 concept_ids,
                 run_dir,
                 criteria = 'gender'
                )->None:
        global people, selected_pose, thre
        self.concept_ids = concept_ids
        concept_num = len(concept_ids)
        people = people
        selected_pose = selected_pose
        assert len(concept_ids)==len(selected_pose)
        #run_dir = '../sefa/drift_data/pose2'
        #run_dir = './download/pose2'
        run_dir = os.path.join(run_dir,'download/pose2')
        postfix = 'jpg'
        thre = thre
        with open(os.path.join(run_dir, 'pred','10000logits.pkl'), 'rb') as f:
            logits = pickle.load(f)
        with open(os.path.join(run_dir, 'pred','10000labels.pkl'), 'rb') as f:
            labels = pickle.load(f)
        logits = np.array(logits[criteria])
        labels = np.array(labels[criteria])
        import pandas as pd
        conf_mask = logits>thre
        se = pd.Series(labels, dtype="category")
        self.targets = se.cat.codes.values
        self.t_mapping = dict(enumerate(se.cat.categories ))
        self.targets = self.targets[conf_mask]
        self.targets = np.expand_dims(self.targets, axis=0).repeat(len(selected_pose), axis=0).flatten()
        self.data = []
        for pose_id in selected_pose:
            per_pose = []
            for pic_id in range(people):
                per_pose.append(os.path.join(run_dir, f'{pic_id}_{pose_id}.{postfix}'))
            per_pose = np.array(per_pose)[conf_mask].tolist()
            self.data.extend(per_pose)
#         self.data = np.array([os.path.join(run_dir, f'{pic_id}_{pose_id}.{postfix}') \
#                      for pose_id in selected_pose for pic_id in range(people)]) #store file name
        self.data = np.array(self.data)
        assert len(self.data)==len(self.targets), 'Mismatch between data and targets len'
        item_num = len(self.data)//concept_num
        self.concept_indices = np.array\
        ([[_id]*item_num for _id,_ in enumerate(range(0,len(self.data),item_num))])
        if len(self.data)%concept_num!=0:
            self.concept_indices[-1] = self.concept_indices[-2][0]
        self.concept_indices = self.concept_indices.flatten().astype(np.int32)[:len(self.data)]
    def __len__(self):
        return len(self.data)