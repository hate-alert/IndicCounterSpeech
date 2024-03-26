import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import torch
from tqdm import tqdm
from nltk import word_tokenize
import nltk
from nltk.translate import meteor
from nltk.translate.bleu_score import SmoothingFunction

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

import re
import io
import os
from glob import glob
from bert_score import score

import warnings
warnings.filterwarnings('ignore')

##########################        Pre-processing        #####################################

def processText(text):
    text = re.sub(r"\S*https?:\S*", "", text)
    #text = re.sub('<user>','',text)
    #text = re.sub('<url>','',text)
    text = re.sub('<.*?>','',text)
    text = re.sub(r'[.!"\/<\*>!@#$%^&*]', r'', text)
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", '', text)
    text = re.sub(' +', ' ', text)
    _RE_COMBINE_WHITESPACE = re.compile(r"(?a:\s+)")
    _RE_STRIP_WHITESPACE = re.compile(r"(?a:^\s+|\s+$)")
    text = _RE_COMBINE_WHITESPACE.sub(" ", text)
    text = _RE_STRIP_WHITESPACE.sub("", text)
    text = text.strip()
    return text


def remove_emojis(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f" 
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)

def preprocess(text):
    text = remove_emojis(text)
    text = processText(text)
    return text



#############################    METRICS   #############################

def hate_refrences(data,test_set):          ###############returns pair of <hate,refrences>  
    hate  = []
    reply = []
    refrences = []
    for sample in data:
        ht , rep = sample[0] , sample[1]
        hate.append(ht)
        reply.append(rep)
    hate = list(set(hate))
    mp={}
    for ht_i in hate:
        refs = []
        for sample in data:
            ht_j , rep =  sample[0] , sample[1]
            if ht_j == ht_i:
                refs.append(rep)
        mp[ht_i] = refs
        refrences.append(refs)
    hate = list(set([x[0] for x in test_set]))
    refs = [mp[ht_i] for ht_i in hate]
    return hate,refs             # a given hate instance and refrences(replies) for metrics evaluation


# In[7]:


def training_corpus(train_set):    # returns training corpus
    replies = []
    for sample in train_set:
        rep = sample[1]
        replies.append(rep)
    replies = list(set(replies))
    return replies                # returns the sentences used while training 




def evaluate(params, model, test_dataloader, device):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
        inputs, labels = (batch[0], batch[0])
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
        
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    return perplexity


###################################### BLEU_SCORE , METEOR #######################################
def hate_refrences(data, test_set):          ###############returns pair of <hate,refrences>  
    hate  = []
    reply = []
    refrences = []
    for ind in data.index:
        ht , rep = data['input_text'][ind] , data['target_text'][ind]
        hate.append(ht)
        reply.append(rep)
    hate = list(set(hate))
    mp={}
    for ht_i in hate:
        refs = []
        for ind in data.index:
            ht_j , rep =  data['input_text'][ind] , data['target_text'][ind]
            if ht_j == ht_i:
                refs.append(rep)
        mp[ht_i] = refs
        refrences.append(refs)
    #hate = list(set([x[0] for x in test_set]))
    #refs = [mp[ht_i] for ht_i in hate]
    return hate, refrences   



############################################ JACCARD SIMILARITY #################################
def get_jaccard_sim(str1, str2):   
    if isinstance(str1, float) or isinstance(str2, float):
        return (-1)
    try:
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        print((str1))
        print(type(str2))
        return 0


############################################### NOVELTY #########################################
def get_novelty(sent, training_corpus):
    max_overlap = 0
    for instance in training_corpus:
        max_overlap = max(max_overlap,get_jaccard_sim(instance,sent))
    return 1-max_overlap

def avg_novelty(sentences,training_corpus):
    avg = 0
    for sent in sentences:
        avg += get_novelty(sent,training_corpus)
    avg = (avg/float(len(sentences)))
    return avg



############################################### DIVERSITY ########################################
def get_diversity(sentences):
    avg = 0.0
    for i in range(len(sentences)):
        max_overlap = 0
        for j in range(len(sentences)):
            if i!=j:
                max_overlap = max(max_overlap,get_jaccard_sim(sentences[i],sentences[j]))
        avg = avg + (1-max_overlap)
    avg = (avg/len(sentences))
    return avg, len(sentences)
    
def diversity_and_novelty(training_corpus, gen_replies):
    diversity = get_diversity(gen_replies)
    novelty   = 0#avg_novelty(gen_replies,training_corpus)
    return diversity,novelty

## bleu and meteor scores
def get_references(df_train, df_test):
    hate  = []
    reply = []
    refrences = []
    for ind in df_train.index:
        ht , rep = df_train['input_text'][ind] , df_train['target_text'][ind]
        hate.append(ht)
        reply.append(rep)

    for ind in df_test.index:
        ht , rep = df_test['input_text'][ind] , df_test['target_text'][ind]
        hate.append(ht)
        reply.append(rep)

    hate = list(set(hate))
    mp={}

    for ht_i in hate:
        refs = []
        for ind in df_train.index:
            ht_j , rep =  df_train['input_text'][ind] , df_train['target_text'][ind]
            if ht_j == ht_i:
                refs.append(rep)
        for ind in df_test.index:
            ht_j , rep =  df_test['input_text'][ind] , df_test['target_text'][ind]
            if ht_j == ht_i:
                refs.append(rep)
        mp[ht_i] = refs
        refrences.append(refs)
        
    return mp



###  Calculate BERT Score ###
def bert_score(df_pred, mp, language):
    bs_p = bs_r = bs_f1 = 0.0
    cands = []
    for i, ind in tqdm(enumerate(df_pred.index), total=len(df_pred)):
        hates = df_pred['input_text'][ind]
        counters = df_pred['predicted_text'][ind]
        ref = mp[hates]

        ref_list = []
        cands = []
        for i in range(len(ref)):
            ref_list.append(ref[i])

        pre, rec, f1 = score([counters], [ref_list], device='cuda:1', lang=language, rescale_with_baseline=True)
        bs_p+=pre
        bs_r+=rec
        bs_f1+=f1
        #print(pre, rec, f1)

    bs_p /= len(df_pred)
    bs_r /= len(df_pred)
    bs_f1 /= len(df_pred)

    return bs_p, bs_r, bs_f1
    

def bert_score_main(df_train, df_test, df_pred, language):    
#     if 'hindi' in pred_path:
#         #df_train = pd.read_csv(train_path) #,lineterminator='\n')
#         #df_pred = pd.read_csv(pred_path) #,lineterminator='\n')
#         df_pred = df_pred.fillna('')
#         #df_test = pd.read_csv(test_path) #,lineterminator='\n')
#     else:
#         df_train = pd.read_csv(train_path, lineterminator='\n')
#         df_pred = pd.read_csv(pred_path, lineterminator='\n')
#         df_pred = df_pred.fillna('')
#         df_test = pd.read_csv(test_path, lineterminator='\n')
    
    df_pred = df_pred.fillna('')
    for ind in df_pred.index:
        df_pred['input_text'][ind] =  preprocess(df_pred['input_text'][ind])
        df_pred['predicted_text'][ind] =  preprocess(df_pred['predicted_text'][ind])

    for ind in df_train.index:
        df_train['input_text'][ind] =  preprocess(df_train['input_text'][ind])
        df_train['target_text'][ind] =  preprocess(df_train['target_text'][ind])

    for ind in df_test.index:
        df_test['input_text'][ind] =  preprocess(df_test['input_text'][ind])
        df_test['target_text'][ind] =  preprocess(df_test['target_text'][ind])
            
    mp = get_references(df_train, df_test)
    bs_p, bs_r, bs_f1 = bert_score(df_pred, mp, language)
    #print('{} : {}, {}, {}'.format(pred_path.split('/')[-1].split('.')[0], bs_p, bs_r, bs_f1))
    return bs_p, bs_r, bs_f1
        
'''
if __name__ == '__main__':
    
    base_path = '/home/mithun-binny/HateAlert_Folder/JointDir/Saurabh/'
    
    train_hindi_path = base_path + 'data_final/Exp1/Hindi/hindi_train_pairs.csv'
    test_hindi_path = base_path + 'data_final/Exp1/Hindi/hindi_test_pairs.csv'
    
    train_bengali_path = base_path + 'data_final/Exp1/Bengali/bengali_train_pairs.csv'
    test_bengali_path = base_path + 'data_final/Exp1/Bengali/bengali_test_pairs.csv'
    
    
    PATH = '/home/mithun-binny/HateAlert_Folder/JointDir/Saurabh/outputs/Exp1'
    EXT = "*.csv"
    all_csv_files = [file
                     for path, subdir, files in os.walk(PATH)
                     for file in glob(os.path.join(path, EXT))]
    for pred_path in all_csv_files:
        f1 = open('/home/mithun-binny/HateAlert_Folder/JointDir/Saurabh/evaluation/results.txt', 'a')
        print("Running on ", pred_path)
        if 'hindi' in pred_path:
            bs_p, bs_r, bs_f1 = main(train_hindi_path, test_hindi_path, pred_path)
        else:
            bs_p, bs_r, bs_f1 = main(train_bengali_path, test_bengali_path, pred_path)
        print(bs_p, bs_r, bs_f1)
        print("Done and writing in file")
        f1.write(pred_path.split('/')[-1].split('.')[0] + ' : ' + ' ' + str(round(float(bs_p), 4)) + ' ' + str(round(float(bs_r), 4)) + ' ' + str(round(float(bs_f1), 4)))
        f1.write('\n')
        print("Written")
        f1.close()
'''