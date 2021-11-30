
###################


import re
from flask import json



####################
import pandas as pd

#from flair.models import SequenceTagger #### 
#model_s = SequenceTagger.load('ner-ontonotes-fast') #.load('ner')
#from flair.data import Sentence
import spacy   
sp_lg = spacy.load('en_core_web_lg')
nlp_c= spacy.load('en_core_web_lg')

    
def expand_contractions(s):
    contractions_dict = json.load(open("dct_exapand.txt"))
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)
    


def replace_NE_date(sentence): ## replaces numbers with the constatnt token "cardinal"    
    sentence=replace_ne(sentence)
    sentence=replace_date_cardinal(sentence)    
    sentence=re.sub("(\d+)","cardinal ",sentence)
    return(sentence)

def replace_ne(sentence): ## third party model to recognize addresse related NE
    s = Sentence(sentence)
    model_s.predict(s)
    results=s.to_dict(tag_type='ner')
    for result in  results['entities']:
        rs=str(result['labels'][0]).split()
        sentence=sentence.replace(result['text'],rs[0])
    return sentence

def replace_date_cardinal(sentence): # spacy based date and NE recognition
    results={(ent.text.strip(), ent.label_) for ent in sp_lg(sentence).ents}
    if results:
        for result in results:
            sentence=sentence.replace(result[0],result[1])
        return sentence
    return sentence

def pre_process(path): # calls the date,entity recognizers to normalizes the texts
    global df_original
    preprocessed_sentences=[]
    df=pd.read_csv(path)
    #df = df.dropna()
    #df=df.sample(frac=1)
    df_original=df
    original_sentences=df.response_text.values
    intents=df.intent.values
    for sentence in original_sentences:
        #print(sentence)
        sentence=sentence.replace(",",", ")
        preprocessed_sentences.append(replace_NE_date(sentence))
    return preprocessed_sentences,original_sentences,intents,df_original