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
        
        if(parser.getboolean('generate', 'generate_hindi')==True):
            print("Generating Hindi")
            self.model_name = parser['models']['model_name_hindi']
            self.test_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_test_hindi'], encoding='utf-8')
            self.model = torch.load(parser['paths']['base_path']+parser['models']['saved_model_gpt2_hindi'])
            self.output_path = parser['paths']['base_path']+parser['outputs']['output_hindi']
        else:
            print("Generating Bengali")
            self.model_name = parser['models']['model_name_bengali']
            self.test_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_test_bengali'], encoding='utf-8') 
            self.model = torch.load(parser['paths']['base_path']+parser['models']['saved_model_gpt2_bengali'])
            self.output_path = parser['paths']['base_path']+parser['outputs']['output_bengali']
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({"eos_token": "</s>", 
                           "bos_token": "<s>", 
                           "unk_token": "<unk>", 
                           "pad_token": "<pad>", 
                           "mask_token": "<mask>",
                           "sep_token": " <EOS> <BOS> "})
        self.training = True
        self.pred_data = pd.DataFrame(columns = ['prefix','input_text','predicted_text'])
        self.dev = torch.device("cuda:{}".format(parser['train']['gpu_num']) 
                                if torch.cuda.is_available() else torch.device("cpu"))
        self.min_length = int(parser['generate']['min_length'])
        self.max_length = int(parser['generate']['max_length'])
        self.sample = parser.getboolean('generate','sample')

    def generate_gpt2(self):
        self.model.eval()
        print(".....Model checkpoint loaded....")
        outputs = []
        num_data =  len(self.test_data)
        for i in tqdm(list(range(num_data)), desc="Generating Counter: "):
            result = {}
            input_ids = self.tokenizer.encode(preprocess(self.test_data.iloc[i]['input_text'])+'পাল্টা বক্তব্য',
                                              return_tensors="pt")
            input_ids=input_ids.to(self.dev)
            output = self.model.generate(input_ids=input_ids, 
                                         max_length=self.max_length,
                                         min_length=self.min_length,
                                         do_sample=self.sample,
                                         pad_token_id = self.tokenizer.eos_token_id)
            pred_text = self.tokenizer.decode(output[0])
            print(pred_text)
            print(pred_text.split('পাল্টা বক্তব্য')[-1])
            result['prefix'] = 'counterspeech'
            result['input_text'] = preprocess(self.test_data.iloc[i]['input_text'])
            result['target_text'] = preprocess(self.test_data.iloc[i]['target_text'])
            result['predicted_text'] = pred_text.split('পাল্টা বক্তব্য')[-1]
            outputs.append([result])
        json.dump(outputs, open(self.output_path, 'w'), indent = 4, ensure_ascii=False)
        print("...Saved outputs...")
        

if __name__ == '__main__':
    path = '/home/mithun-binny/HateAlert_Folder/JointDir/Saurabh/model_code/gpt2_bengali/config.cfg'
    Counterspeech_generate(path).generate_gpt2()