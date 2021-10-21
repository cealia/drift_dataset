import pandas as pd
import os

class Review_y:
    def __init__(self, 
                 concept_ids,
                 run_dir
                )->None:
        run_dir = os.path.join(run_dir,'download/review')
        df = pd.read_csv(os.path.join(run_dir, 'final.csv'))
        
        self.concept_ids = concept_ids
        concept_num = len(concept_ids)
        
        self.data = df.filter(like='un_', axis=1).to_numpy()
        self.targets = df.for_y_label.to_numpy()
        self.concept_indices = df.for_y.to_numpy()

        
    def __len__(self):
        return len(self.data)