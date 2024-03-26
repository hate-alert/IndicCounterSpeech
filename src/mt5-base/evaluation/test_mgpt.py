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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5Tokenizer, TextGenerationPipeline
# from transformers import BloomTokenizerFast, BloomModel, BloomForCausalLM
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import configparser
from pre_process import preprocess
from eval_metrics import eval_main


class Counterspeech_generate():

    def __init__(self, path, index):
        parser=configparser.ConfigParser()
        parser.read(path)
        
        self.train_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_train'], 
                                      encoding='utf-8',lineterminator='\n')
        self.test_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_test'], 
                                     encoding='utf-8',lineterminator='\n')
        self.pred_data = pd.DataFrame(columns = ['prefix','input_text','predicted_text'])
        
        self.model_name = parser['models']['model_name']
        self.dev = torch.device("cuda:{}".format(parser['models']['gpu_num']) 
                                if torch.cuda.is_available() else torch.device("cpu"))
        
        self.model = torch.load(parser['paths']['base_path']+parser['models']['saved_model']+'_'+str(index)+'.pt',
                                map_location=self.dev)
        self.tokenizer = MT5Tokenizer.from_pretrained(self.model_name)
        #self.tokenizer = BloomTokenizerFast.from_pretrained(self.model_name)
        
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer,device=int(parser['models']['gpu_num']))
        self.output_path_json = parser['paths']['base_path']+parser['outputs']['output_json']+'_'+str(index)+'.json'
        self.output_path_csv = parser['paths']['base_path']+parser['outputs']['output_csv']+'_'+str(index)+'.csv'
        self.batch_size = int(parser['models']['batch_size'])
        
    def generate_gpt2(self):
        self.model.eval()
        print(".....GPT2 Model checkpoint loaded....")
        outputs = []
        num_data =  len(self.test_data)
        for i in tqdm(list(range(num_data)), desc="Generating Counter: "):
            result = {}
            input_ids = preprocess(self.test_data.iloc[i]['input_text'])
            #input_ids = self.tokenizer.encode(preprocess(self.test_data.iloc[i]['input_text']),return_tensors="pt")
            #input_ids=input_ids.to(self.dev)
            with torch.no_grad():
                output = self.pipeline(input_ids,
                                       max_length=1024,
                                       do_sample = True)
            #pred_text = self.tokenizer.decode(output[0])
            pred_text = output[0]['generated_text']
            result['prefix'] = 'counterspeech'
            result['input_text'] = preprocess(self.test_data.iloc[i]['input_text'])
            result['target_text'] = preprocess(self.test_data.iloc[i]['target_text'])
            result['predicted_text'] = pred_text.split(result['input_text'])[-1]
            outputs.append([result])
            self.pred_data = self.pred_data.append(result, ignore_index=True)
        json.dump(outputs, open(self.output_path_json, 'w'), indent = 4, ensure_ascii=False)
        self.pred_data.to_csv(self.output_path_csv, index=False)
        print("...Saved outputs...")
        
    def get_scores(self):
#         ppl = self.evaluate_test()
#         print("Perplexity: ", ppl)
        eval_main(self.train_data, self.test_data, self.pred_data)
        
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
    path = '/home/mithundas/HateAlert_Folder/JointDir/Saurabh/evaluation/config_9191.cfg'
    cs = Counterspeech_generate(path)
    cs.generate_gpt2()
    cs.get_scores()