import pandas as pd
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import  Adafactor 
import time
import warnings
warnings.filterwarnings('ignore')
import urllib.request
import zipfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import configparser
from utils import preprocess

class Counterspeech_generate():

    def __init__(self, path):
        parser=configparser.ConfigParser()
        parser.read(path)
        
        if(parser.getboolean('train', 'train_hindi')==True):
            self.model_name = parser['models']['model_name_hindi']
            self.test_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_test_hindi'], encoding='utf-8')
            self.model = torch.load(parser['paths']['base_path']+parser['models']['saved_model_gpt2_hindi'])
            self.output_path = parser['paths']['base_path']+parser['outputs']['output_hindi']
        else:
            self.model_name = parser['models']['model_name_bengali']
            self.test_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_test_bengali'], encoding='utf-8') 
            self.model = torch.load(parser['paths']['base_path']+parser['models']['saved_model_mt5_bengali'])
            self.output_path = parser['paths']['base_path']+parser['outputs']['output_bengali']
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.training = True
        self.pred_data = pd.DataFrame(columns = ['prefix','input_text','predicted_text'])
        self.dev = torch.device("cuda:{}".format(parser['train']['gpu_num']) 
                                if torch.cuda.is_available() else torch.device("cpu"))

    def generate(self):
        self.model.eval()
        print(".....Model checkpoint loaded....")
        outputs = []
        num_data =  len(self.test_data)
        for i in tqdm(list(range(num_data)), desc="Generating Counter: "):
            result = {}
            input_ids = self.tokenizer.encode(preprocess(self.test_data.iloc[i]['input_text']),return_tensors="pt")
            input_ids=input_ids.to(self.dev)
            print("Predicting")
            output = self.model.generate(input_ids=input_ids, 
                                         max_length=400,
                                         min_length=20,
                                         do_sample=True)
            pred_text = self.tokenizer.decode(output[0])
            print(pred_text)
            result['prefix'] = 'counterspeech'
            result['input_text'] = self.test_data.iloc[i]['input_text']
            result['predicted_text'] = pred_text
            outputs.append([result])
        json.dump(outputs, open(self.output_path, 'w'), indent = 4, ensure_ascii=False)
        print("...Saved outputs...")
        

if __name__ == '__main__':
    path = '/home/mithun-binny/HateAlert_Folder/JointDir/Saurabh/model_code/gpt2_suraj/config.cfg'
    Counterspeech_generate(path).generate()