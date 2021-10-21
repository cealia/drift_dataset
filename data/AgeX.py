import os
import pandas as pd


class AgeX:
    def __init__(self,
                 concept_ids,
                 run_dir,
                )->None:
        run_dir = os.path.join(run_dir,'download/AgeX')
        df = pd.read_csv(os.path.join(run_dir, 'meta1_3.csv'))
        #print('meta1_1.csv')
        self.concept_ids = sorted(list(set(df.concept_id.values.astype(int))))
        assert concept_ids==self.concept_ids
        df['add_run'] = df.name.apply(lambda x: os.path.join(run_dir, x))
        
        self.data = df.add_run.values #store filename
        self.targets = df.gender.values
        self.concept_indices = df.concept_id.values.astype(int)
        
    def __len__(self):
        return len(self.data)