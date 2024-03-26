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
from transformers import BloomTokenizerFast, BloomModel, BloomForCausalLM
import torch.nn.functional as F


class Counterspeech_generate():

    def __init__(self, path):
        parser=configparser.ConfigParser()
        parser.read(path)
        
        if(parser.getboolean('generate', 'generate_hindi')==True):
            print("Generating Hindi")
            self.model_name = parser['models']['model_name_hindi']
            self.test_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_test_hindi'], encoding='utf-8')
            self.model = torch.load(parser['paths']['base_path']+parser['models']['saved_model_bloom_hindi'])
            self.output_path = parser['paths']['base_path']+parser['outputs']['output_hindi']
        else:
            print("Generating Bengali")
            self.model_name = parser['models']['model_name_bengali']
            self.test_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_test_bengali'], encoding='utf-8') 
            self.model = torch.load(parser['paths']['base_path']+parser['models']['saved_model_bloom_bengali'])
            self.output_path = parser['paths']['base_path']+parser['outputs']['output_bengali']
            
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.model_name)
        self.training = True
        self.pred_data = pd.DataFrame(columns = ['prefix','input_text','predicted_text'])
        #self.dev = torch.device("cpu")
        self.dev = torch.device("cuda:{}".format(parser['train']['gpu_num']) 
                                if torch.cuda.is_available() else torch.device("cpu"))
        self.max_length = int(parser['generate']['max_length'])
        self.min_length = int(parser['generate']['min_length'])
        self.sample = parser.getboolean('generate','sample')

    def generate_bloom(self):
        self.model.to(self.dev)
        self.model.eval()
        print(".....Model checkpoint loaded....")
        outputs = []
        num_data =  len(self.test_data)
        for i in tqdm(list(range(num_data)), desc="Generating Counter: "):
            result = {}
            input_ids = self.tokenizer.encode(self.test_data.iloc[i]['input_text'],
                                              return_tensors="pt")
            input_ids=input_ids.to(self.dev)
            output = self.model.generate(input_ids=input_ids, 
                                         max_length=self.max_length,
                                         min_length=self.min_length,
                                         do_sample=self.sample)
            pred_text = self.tokenizer.decode(output[0])
            result['prefix'] = 'counterspeech'
            result['input_text'] = self.test_data.iloc[i]['input_text']
            result['predicted_text'] = pred_text
            outputs.append([result])
        json.dump(outputs, open(self.output_path, 'w'), indent = 4, ensure_ascii=False)
        print("...Saved outputs...")
        

if __name__ == '__main__':
    path = '/home/mithun-binny/HateAlert_Folder/JointDir/Saurabh/model_code/bloom/config.cfg'
    Counterspeech_generate(path).generate_bloom()