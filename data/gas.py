import pandas as pd
import numpy as np
import os


class gas:
    def __init__(self, 
                 concept_ids,
                 run_dir
                )->None:
        
        run_dir = os.path.join(run_dir,'download/gas')
        #df = pd.read
        
        ls = []
        for _id,i in enumerate(range(1,11)):
            fn = os.path.join(run_dir, f'batch{i}.dat')
            df = pd.read_csv(fn, sep=' ', header=None)
            #print(len(df))
            #print(df[0].value_counts())
            df = df.apply(lambda x: x.apply(lambda y:np.float(y.split(':')[-1]) if isinstance(y,str) else y))
            df['concept_indices'] = _id
            ls.append(df)
        df = pd.concat(ls, ignore_index=True)
        df.rename(columns={0: 'label'}, inplace=True)
        df.label = df.label-1
        def normalize(x):
            #print(x)
            temp = x.loc[:, list(range(1,128+1))]
            new = (temp-temp.mean())/temp.std()
            new['label'] = x.label
            new['concept_indices'] = x.concept_indices
            return new
        df = df.groupby(by=['concept_indices']).apply(normalize) #.apply(lambda x: print(x))#
        
        self.concept_ids = concept_ids
        self.data = df.loc[:, list(range(1,128+1))].to_numpy()
        self.targets = df['label'].values
        self.concept_indices = df['concept_indices'].values
        
        
    def __len__(self):
        return len(self.data)