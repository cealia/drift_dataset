import pandas as pd
import numpy as np
import os

class Covertype:
    def __init__(self, 
                 concept_ids,
                 run_dir
                )->None:
        run_dir = os.path.join(run_dir,'download/covertype')
        df = pd.read_csv(os.path.join(run_dir, 'covertype_csv.csv'))
        
        self.concept_ids = concept_ids
        concept_num = len(concept_ids)
        
        self.data = df.iloc[:,:len(df.columns)-1].to_numpy()
        self.targets = df.iloc[:,-1].to_numpy()-1 #change to 0-index
        
        item_num = len(df)//concept_num
        self.concept_indices = np.array([[_id]*item_num for _id,_ in enumerate(range(0,len(df),item_num))])
        self.concept_indices[-1] = self.concept_indices[-2][0]
        self.concept_indices = self.concept_indices.flatten().astype(np.int32)[:len(df)]
        
        
    def __len__(self):
        return len(self.data)