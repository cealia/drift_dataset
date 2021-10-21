import pandas as pd
import os
from collections import Counter
import numpy as np

class Review_x:
    def __init__(self, 
                 concept_ids,
                 run_dir
                )->None:
        run_dir = os.path.join(run_dir,'download/review')
        df = pd.read_csv(os.path.join(run_dir, 'final.csv'))
        
        self.concept_ids = concept_ids
        concept_num = len(concept_ids)
        
        self.data = df.filter(like='un_', axis=1).to_numpy()
        
        temp = df.overall.to_numpy()
        self.targets = np.zeros(len(temp), dtype=int)
        self.targets[temp>2] = 1
        print(Counter(self.targets))
        
        self.concept_indices = df.id.to_numpy()

        
    def __len__(self):
        return len(self.data)