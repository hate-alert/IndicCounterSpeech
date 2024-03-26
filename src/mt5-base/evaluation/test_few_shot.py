import numpy as np
import pandas as pd
from test_fs import Counterspeech_generate
import configparser

def main(path):
    for i in range(0,3):
        print("###### Evaluating Model {} ######".format(i+1))
        cs = Counterspeech_generate(path, i)
        cs.generate_mt5()
        cs.get_scores()
        
if __name__ == '__main__':
    path = '/home/mithun-binny/HateAlert_Folder/JointDir/Saurabh/model_code/mT5_base/evaluation/config_few_shot.cfg'
    parser=configparser.ConfigParser()
    parser.read(path)
    main(path)