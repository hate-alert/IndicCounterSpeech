import numpy as np
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
import numpy as np
import pandas as pd
import re
import io
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.translate.meteor_score import meteor_score
from eval_scores import *
from pre_process import preprocess

def eval_main(df_train, df_test, df_pred, output_file):
    f = open(output_file,'w')
    
    for ind in df_pred.index:
        df_pred['input_text'][ind] =  preprocess(df_pred['input_text'][ind])
        df_pred['predicted_text'][ind] =  preprocess(df_pred['predicted_text'][ind])

    for ind in df_train.index:
        df_train['input_text'][ind] =  preprocess(df_train['input_text'][ind])
        df_train['target_text'][ind] =  preprocess(df_train['target_text'][ind])

    for ind in df_test.index:
        df_test['input_text'][ind] =  preprocess(df_test['input_text'][ind])
        df_test['target_text'][ind] =  preprocess(df_test['target_text'][ind])


    ## Diversity Scores
    #print("Diversity Scores")
    f.write("Diversity Scores")
    #print(get_diversity(df_train['input_text']))
    #print(get_diversity(df_train['target_text']))
    f.write("\nTrain input text, Train target text : " + str(get_diversity(df_train['input_text']))
            + str(get_diversity(df_train['target_text'])))

    #print(get_diversity(df_test['input_text']))
    #print(get_diversity(df_test['target_text']))
    f.write("\nTest input text, Test target text : " + str(get_diversity(df_test['input_text'])) +
            str(get_diversity(df_test['target_text'])))

    #print(get_diversity(df_pred['input_text']))
    #print(get_diversity(df_pred['predicted_text']))
    f.write("\nPredicted input text, Predicted pred text : " +  str(get_diversity(df_pred['input_text'])) + 
            str(get_diversity(df_pred['predicted_text'])))


    ## Novelty Scores
#     print("Novelty Scores")
#     print(avg_novelty(df_train['input_text'], df_train['input_text']), avg_novelty(df_train['input_text'], df_train['target_text']))
#     print(avg_novelty(df_train['input_text'], df_test['input_text']), avg_novelty(df_train['input_text'], df_test['target_text']))
#     print(avg_novelty(df_train['input_text'], df_pred['input_text']), avg_novelty(df_train['input_text'], df_pred['predicted_text']))

#     print(avg_novelty(df_train['target_text'], df_train['input_text']), avg_novelty(df_train['target_text'], df_train['target_text']))
#     print(avg_novelty(df_train['target_text'], df_test['input_text']), avg_novelty(df_train['target_text'], df_test['target_text']))
#     print(avg_novelty(df_train['target_text'], df_pred['input_text']), avg_novelty(df_train['target_text'], df_pred['predicted_text']))

#     print(avg_novelty(df_test['input_text'], df_train['input_text']), avg_novelty(df_test['input_text'], df_train['target_text']))
#     print(avg_novelty(df_test['input_text'], df_test['input_text']), avg_novelty(df_test['input_text'], df_test['target_text']))
#     print(avg_novelty(df_test['input_text'], df_pred['input_text']), avg_novelty(df_test['input_text'], df_pred['predicted_text']))

#     print(avg_novelty(df_test['target_text'], df_train['input_text']), avg_novelty(df_test['target_text'], df_train['target_text']))
#     print(avg_novelty(df_test['target_text'], df_test['input_text']), avg_novelty(df_test['target_text'], df_test['target_text']))
#     print(avg_novelty(df_test['target_text'], df_pred['input_text']), avg_novelty(df_test['target_text'], df_pred['predicted_text']))

#     print(avg_novelty(df_pred['predicted_text'], df_train['input_text']), avg_novelty(df_pred['predicted_text'], df_train['target_text']))
#     print(avg_novelty(df_pred['predicted_text'], df_test['input_text']), avg_novelty(df_pred['predicted_text'], df_test['target_text']))
#     print(avg_novelty(df_pred['predicted_text'], df_pred['input_text']), avg_novelty(df_pred['predicted_text'], df_pred['predicted_text']))


#     print("Novelty Score between 1) predicted and test counters 2) predicted and train counters")
#     print(avg_novelty(df_pred['predicted_text'], df_test['target_text']), avg_novelty(df_pred['predicted_text'], df_train['target_text']))
    
    f.write("\n\n")
    f.write("Novelty scores")
    f.write("\nPredicted pred text, Test target text : " + str(avg_novelty(df_pred['predicted_text'], df_test['target_text'])))
    f.write("\nPredicted pred text, Train target text : " + str(avg_novelty(df_pred['predicted_text'], df_train['target_text'])))

    ## bleu and meteor scores
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

    bleu = bleu_2 = meteor_ = 0.0

    for ind in df_pred.index:
        hates = df_pred['input_text'][ind]
        counters = df_pred['predicted_text'][ind]
        ref = mp[hates]

        ref_list = []
        for i in range(len(ref)):
            ref_list.append(word_tokenize(ref[i]))
        bleu += nltk.translate.bleu_score.sentence_bleu(ref_list, word_tokenize(counters))
        bleu_2  += nltk.translate.bleu_score.sentence_bleu(ref_list, word_tokenize(counters),
                                                           smoothing_function=SmoothingFunction().method2, weights=(0.5, 0.5, 0, 0))
        meteor_ += meteor_score(ref_list, word_tokenize(counters))

    bleu    /= len(df_pred)
    bleu_2  /= len(df_pred)
    meteor_ /= len(df_pred)

#     print("Bleu Score ", bleu)
#     print("Bleu Score with Smoothener", bleu_2)
#     print("Meteor Score", meteor_)
    
    f.write("\n\n")
    f.write("\nBleu Score : " + str(bleu))
    f.write("\nBleu Score with Smoothener : " + str(bleu_2))
    f.write("\nMeteor Score : " + str(meteor_))


    def rec(str1, str2):
        match = 0.0
        tok1 = word_tokenize(str1)
        tok2 = word_tokenize(str2)
        if(len(tok1)==0 or len(tok2)==0):
            return -999
        for i in tok1:
            for j in tok2:
                if i == j:
                    match += 1.0
                    break;
        return match/len(tok1)

    def rec2(str1, str2):
        match = 0.0
        tok1 = word_tokenize(str1)
        tok2 = word_tokenize(str2)
        for i in tok2:
            for j in tok1:
                if i == j:
                    match += 1.0
                    break;
        return match/len(tok2)

    recall = 0.0

    for ind in df_pred.index:
        recall2 = 0.0
        hates = df_pred['input_text'][ind]
        counters = df_pred['predicted_text'][ind]
        ref = mp[hates]

        for i in range(len(ref)):
            recall2 = max(recall2, rec(counters, ref[i]))
            #print(recall2)

        recall += recall2

    recall    /= len(df_pred)



    precision = 0.0

    for ind in df_pred.index:
        recall2 = 0.0
        hates = df_pred['input_text'][ind]
        counters = df_pred['predicted_text'][ind]
        ref = mp[hates]

        for i in range(len(ref)):
            recall2 = max(recall2, rec2(counters, ref[i]))
            #print(recall2)

        precision += recall2

    precision    /= len(df_pred)

#     print("Precision: " + str(precision))
#     print("Recall: " + str(recall))
#     print("F-score: " + str(2*precision*recall/(precision+recall)))
#     print("...Metrics calculated...")
    
    f.write("\n\n")
    f.write("\nPrecision : " + str (precision))
    f.write("\nRecall : " + str(recall))
    f.write("\nF-score : " + str(2*precision*recall/(precision+recall)))
    
    f.close()