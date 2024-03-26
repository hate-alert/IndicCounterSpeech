import numpy as np
import pandas as pd
from train_fs import Counterspeech_train
import configparser

def main(path, df):
    for i in range(0,3):
        print("########## Running Model {} ##########".format(i+1))
        df_train = df.sample(200,random_state=40+i)
        Counterspeech_train(path,df_train, i).train_gpt2()
        
if __name__ == '__main__':
    path = './config_few_shot.cfg'
    parser=configparser.ConfigParser()
    parser.read(path)
    df = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_train_bengali'], encoding='utf-8', lineterminator='\n')
    main(path, df)