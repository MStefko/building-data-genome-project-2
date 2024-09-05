# for all csv files in the directory, clean the data and save it to a new csv file
# The new csv file will have the same name as the original file, saved to a new directory
# The new directory will be named 'cleaned_data'

import os
import pandas as pd
import numpy as np
import tqdm


def clean_csv(df, building_name):
    df = df.fillna(0)
    df = df.replace(np.nan, 0)
    df.to_csv(os.path.join(os.path.dirname(__file__),'..','data','buildings_cleaned', f'{building_name}.csv'), index=False)
    
def main():
    for file in tqdm.tqdm(os.listdir(os.path.join(os.path.dirname(__file__),'..','data','buildings'))):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','data','buildings', file))
            building_name = file.split('.')[0]
            clean_csv(df, building_name)
            
if __name__ == '__main__':
    main()