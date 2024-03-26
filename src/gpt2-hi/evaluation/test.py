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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5Tokenizer
from transformers import BloomTokenizerFast, BloomModel, BloomForCausalLM
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import configparser
from pre_process import preprocess
from eval_metrics import eval_main
from bert_scores import bert_score_main


class Counterspeech_generate():

    def __init__(self, path):
        parser=configparser.ConfigParser()
        parser.read(path)
        print(parser['paths']['base_path'])
        
        self.train_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_train'], 
                                      encoding='utf-8',lineterminator='\n')
        self.test_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_test'], 
                                     encoding='utf-8',lineterminator='\n')
#         self.pred_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_pred'], 
#                                      encoding='utf-8',lineterminator='\n')
        self.pred_data = pd.DataFrame(columns = ['prefix','input_text','predicted_text'])
        
        self.model_name = parser['models']['model_name']
        self.dev = torch.device("cuda:{}".format(parser['models']['gpu_num']) 
                                if torch.cuda.is_available() else torch.device("cpu"))
        
        self.model = torch.load(parser['paths']['base_path']+parser['models']['saved_model'], #+'_'+str(index)+'.pt',
                                map_location=self.dev)
        print("Model laoded : ", parser['paths']['base_path']+parser['models']['saved_model'])
        #self.tokenizer = MT5Tokenizer.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.output_path_json = parser['paths']['base_path']+parser['outputs']['output_json']#+'_'+str(index)+'.json'
        self.output_path_csv = parser['paths']['base_path']+parser['outputs']['output_csv']#+'_'+str(index)+'.csv'
        self.output_path_scores = parser['paths']['base_path']+parser['outputs']['output_scores']
        self.batch_size = int(parser['models']['batch_size'])

    def generate_mt5(self):
        self.model.eval()
        print(".....MT5 Model checkpoint loaded....")
        outputs = []
        num_data =  len(self.test_data)
        for i in tqdm(list(range(num_data)), desc="Generating Counter: "):
            result = {}
            input_ids = self.tokenizer.encode('counterspeech: '+preprocess(self.test_data.iloc[i]['input_text'])+'</s>',
                                              return_tensors="pt")
            input_ids=input_ids.to(self.dev)
            output = self.model.generate(input_ids=input_ids, 
                                         pad_token_id         = self.tokenizer.eos_token_id,
                                         max_length           = 300,
                                         min_length           = 20,
                                         top_k                = 40,
                                         top_p                = 0.92,
                                         repetition_penalty   = 2.0,
                                         num_beams            = 5,
                                         do_sample            = True)
            
            pred_text = self.tokenizer.decode(output[0])
            result['prefix'] = 'counterspeech'
            result['input_text'] = preprocess(self.test_data.iloc[i]['input_text'])
            result['predicted_text'] = pred_text
            outputs.append([result])
            self.pred_data = self.pred_data.append(result, ignore_index=True)
        json.dump(outputs, open(self.output_path_json, 'w'), indent = 4, ensure_ascii=False)
        self.pred_data.to_csv(self.output_path_csv, index=False)
        print("...Saved outputs...")
        
    def generate_gpt2(self):
        self.model.eval()
        print(".....GPT2 Model checkpoint loaded....")
        outputs = []
        num_data =  len(self.test_data)
        for i in tqdm(list(range(num_data)), desc="Generating Counter: "):
            result = {}
            input_ids = self.tokenizer.encode(preprocess(self.test_data.iloc[i]['input_text'])+'</s><s>',
                                              return_tensors="pt")
            input_ids=input_ids.to(self.dev)
            with torch.no_grad():
                output = self.model.generate(input_ids=input_ids, 
                                         pad_token_id         = self.tokenizer.eos_token_id,
                                         max_length           = 300,
                                         min_length           = 20,
                                         top_k                = 40,
                                         top_p                = 0.92,
                                         repetition_penalty   = 2.0,
                                         num_beams            = 5,
                                         do_sample            = True)
            pred_text = self.tokenizer.decode(output[0])
            result['prefix'] = 'counterspeech'
            result['input_text'] = preprocess(self.test_data.iloc[i]['input_text'])
            result['target_text'] = preprocess(self.test_data.iloc[i]['target_text'])
            result['predicted_text'] = pred_text.split('</s><s>')[-1]
            outputs.append([result])
            self.pred_data = self.pred_data.append(result, ignore_index=True)
        json.dump(outputs, open(self.output_path_json, 'w'), indent = 4, ensure_ascii=False)
        self.pred_data.to_csv(self.output_path_csv, index=False)
        print("...Saved outputs...")
        
    def generate_bloom(self):
        self.model.to(self.dev)
        self.model.eval()
        print(".....Bloom Model checkpoint loaded....")
        outputs = []
        num_data =  len(self.test_data)
        for i in tqdm(list(range(num_data)), desc="Generating Counter: "):
            result = {}
            input_ids = self.tokenizer.encode(preprocess(self.test_data.iloc[i]['input_text'])+'</s><s>',
                                              return_tensors="pt")
            input_ids=input_ids.to(self.dev)
            output = self.model.generate(input_ids=input_ids, 
                                         pad_token_id         = self.tokenizer.eos_token_id,
                                         max_length           = 300,
                                         min_length           = 20,
                                         top_k                = 40,
                                         top_p                = 0.92,
                                         repetition_penalty   = 2.0,
                                         num_beams            = 5,
                                         do_sample            = True)
            pred_text = self.tokenizer.decode(output[0])
            result['input_text'] = preprocess(self.test_data.iloc[i]['input_text'])
            result['target_text'] = preprocess(self.test_data.iloc[i]['target_text'])
            result['predicted_text'] = pred_text.split('</s><s>')[-1]
            outputs.append([result])
            self.pred_data = self.pred_data.append(result, ignore_index=True)
        json.dump(outputs, open(self.output_path_json, 'w'), indent = 4, ensure_ascii=False)
        self.pred_data.to_csv(self.output_path_csv, index=False)
        print("...Saved outputs...")
        
    def get_scores(self):
        eval_main(self.train_data, self.test_data, self.pred_data, self.output_path_scores)
        f = open(self.output_path_scores,'a')
        bs_p, bs_r, bs_f1 = bert_score_main(self.train_data, self.test_data, self.pred_data)
        f.write("\n\n\nBERT Scores")
        f.write("\nPrecision : " + str(float(bs_p)))
        f.write("\nRecall : " + str(float(bs_r)))
        f.write("\nF1-Score : " + str(float(bs_f1)))
        f.close()
        print("...Metrics Calculated...")
        
    def evaluate_test(self):
        num_batches=int(len(self.test_data)/self.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()
        self.model.to(self.dev)
        for i in tqdm(list(range(num_batches)), desc="Evaluating testset"):
            inputbatch=[]
            labelbatch=[]
            new_df=self.test_data[i*self.batch_size:i*self.batch_size+self.batch_size]
            for indx,row in new_df.iterrows():
                inputs = 'counterspeech: '+preprocess(row['input_text'])+'</s>' 
                labels = preprocess(row['target_text'])+'</s>'   
                inputbatch.append(inputs)
                labelbatch.append(labels)
            inputbatch=self.tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,
                                                        return_tensors='pt')["input_ids"]
            labelbatch=self.tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,
                                                        return_tensors="pt")["input_ids"]
            inputbatch=inputbatch.to(self.dev)
            labelbatch=labelbatch.to(self.dev)

            with torch.no_grad():
                outputs = self.model(input_ids=inputbatch, labels=labelbatch)
                lm_loss = outputs.loss
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        return perplexity
        
        
if __name__ == '__main__':
    path = './config.cfg'
    cs = Counterspeech_generate(path)
    cs.generate_gpt2()
    cs.get_scores()