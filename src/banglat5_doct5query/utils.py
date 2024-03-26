from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import numpy as np

import re
import io


### Preprocessing functions
def processText(text):
    text = re.sub(r"\S*https?:\S*", "", text)
    #text = re.sub('<user>','',text)
    #text = re.sub('<url>','',text)
    text = re.sub('<.*?>','',text)
    text = re.sub(r'[.!"\/<\*>!@#$%^&*]', r'', text)
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", '', text)
    text = re.sub(' +', ' ', text)
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
    text = processText(text)
    text = remove_emojis(text)
    return text