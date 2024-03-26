import pandas as pd
import numpy as np
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


model_name = 'doc2query/msmarco-hindi-mt5-base-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, return_dict=True)
#model = torch.load('/home/mithun-binny/HateAlert_Folder/JointDir/Saurabh/saved_models/mt5_hindi.pt')
dev = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else torch.device("cpu"))

def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, 103)
    labels = repeat_input.masked_fill( masked_input != 103, -100)
    model = model.to(dev)
    masked_input = masked_input.to(dev)
    labels = labels.to(dev)
    loss = model(masked_input, labels=labels).loss
    return(np.exp(loss.item()))