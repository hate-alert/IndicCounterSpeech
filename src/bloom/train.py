import pandas as pd
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers.optimization import  Adafactor 
import time
import warnings
warnings.filterwarnings('ignore')
import urllib.request
import zipfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from tqdm import tqdm
import configparser
import json
from transformers import BloomTokenizerFast, BloomModel, BloomForCausalLM
import torch.nn.functional as F
from utils import preprocess


class Counterspeech_train():

    def __init__(self, path):
        parser=configparser.ConfigParser()
        parser.read(path)
        
        if(parser.getboolean('train', 'train_hindi')==True):
            print("Training on Hindi")
            self.model_name = parser['models']['model_name_hindi']
            self.train_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_train_hindi'], 
                                          encoding='utf-8',lineterminator='\n')
            self.train_data = self.train_data.sample(frac = 1)
            self.val_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_val_hindi'], 
                                        encoding='utf-8',lineterminator='\n')
            self.val_data = self.val_data.sample(frac = 1)
            self.model = BloomForCausalLM.from_pretrained(self.model_name, return_dict=True)
            self.save_path = parser['paths']['base_path']+parser['models']['save_model_bloom_hindi']
            self.save_final_path = parser['paths']['base_path']+parser['models']['save_model_bloom_hindi_final']
        else:
            print("Training on Bengali")
            self.model_name = parser['models']['model_name_bengali']
            self.train_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_train_bengali'], 
                                          encoding='utf-8',lineterminator='\n')
            self.train_data = self.train_data.sample(frac = 1)
            self.val_data = pd.read_csv(parser['paths']['base_path']+parser['paths']['data_val_bengali'], 
                                        encoding='utf-8',lineterminator='\n')
            self.val_data = self.val_data.sample(frac = 1)
            self.model = BloomForCausalLM.from_pretrained(self.model_name)
            self.save_path = parser['paths']['base_path']+parser['models']['save_model_bloom_bengali']
            self.save_final_path = parser['paths']['base_path']+parser['models']['save_model_bloom_bengali_final']
            
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.model_name)
        self.training = True
        self.dev = torch.device("cuda:{}".format(int(parser['train']['gpu_num'])) 
                                if torch.cuda.is_available() else torch.device("cpu"))
        self.batch_size = int(parser['train']['batch_size'])
        self.num_epochs = int(parser['train']['num_epochs'])
        self.max_length = int(parser['train']['max_length'])
    
    def train_bloom(self):
        num_batches=int(len(self.train_data)/self.batch_size)
        self.model.to(self.dev)

        optimizer = Adafactor(
            self.model.parameters(),
            lr=2e-5,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

        #Sets the module in training mode
        self.model.train()

        loss_per_10_steps=[]
        for epoch in range(1,self.num_epochs+1):
            running_loss=0
            j=0
            for i in tqdm(list(range(num_batches)), desc="Epoch {}".format(epoch)):
                inputs = preprocess(self.train_data.iloc[j]['input_text'])+'</s><s>'+preprocess(self.train_data.iloc[j]\
                                                                                                ['target_text'])+'</s>'
                inputs = self.tokenizer(inputs, return_tensors='pt')
                inputs = inputs.to(self.dev)
                # clear out the gradients of all Variables 
                optimizer.zero_grad()

                # Forward propogation
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss_num=loss.item()
                logits = outputs.logits
                running_loss+=loss_num
                if i%10 ==0:      
                    loss_per_10_steps.append(loss_num)

                # calculating the gradients
                loss.sum().backward()

                #updating the params
                optimizer.step()
                j+=1

            running_loss=running_loss/int(num_batches)
            print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))

            training_ppl = self.evaluate_train()
            validation_ppl = self.evaluate_val()
            print('Epoch: {} , Training PPL: {}'.format(epoch,training_ppl))
            print('Epoch: {} , Validation PPL: {}'.format(epoch,validation_ppl))


            # Save the best model after every epoch
            if(epoch==1):
                best_running_loss=validation_ppl
                torch.save(self.model, self.save_path)
            elif(validation_ppl <= best_running_loss):
                best_running_loss = validation_ppl
                torch.save(self.model, self.save_path)
            print('Epoch: {} , Best Validation PPL: {}'.format(epoch,best_running_loss))

        #Save the final model
        torch.save(self.model,self.save_final_path)
        print("......Model saved....")
            
    def evaluate_val(self):
        num_batches=int(len(self.val_data)/self.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()
        self.model.to(self.dev)
        j=0
        for i in tqdm(list(range(num_batches)), desc="Evaluating Validation"):
            inputs = preprocess(self.val_data.iloc[j]['input_text'])+'</s><s>'+preprocess(self.val_data.iloc[j]['target_text'])+'</s>'
            j+=1
            inputs = self.tokenizer(inputs, return_tensors='pt')
            inputs = inputs.to(self.dev)

            # Forward propogation
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                lm_loss = outputs.loss
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        return perplexity
    
    def evaluate_train(self):
        num_batches=int(len(self.train_data)/self.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()
        self.model.to(self.dev)
        j=0
        for i in tqdm(list(range(num_batches)), desc="Evaluating Training"):
            inputs = preprocess(self.train_data.iloc[j]['input_text'])+'</s><s>'+preprocess(self.train_data.iloc[j]['target_text'])+'</s>'
            j+=1
            inputs = self.tokenizer(preprocess(inputs), return_tensors='pt')
            inputs = inputs.to(self.dev)

            # Forward propogation
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                lm_loss = outputs.loss
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        return perplexity
        
if __name__ == '__main__':
    path = './config.cfg'
    Counterspeech_train(path).train_bloom()