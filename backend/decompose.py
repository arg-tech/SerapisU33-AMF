from typing import ValuesView
from torch.nn import functional as F  
from transformers import pipeline
import pandas as pd
from transformers import AutoTokenizer, AutoModel,BertForTokenClassification,BertTokenizer
import torch
from scipy.special import softmax
import spacy
import numpy as np
import re
from subject_verb_object_extract import findSVOs, printDeps, nlp
import nltk 
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk import pos_tag

from preprocessors import *

nlp = spacy.load("en_core_web_lg")
sbert_tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
sbert_model = AutoModel.from_pretrained('deepset/sentence_bert')
sentiment_model=pipeline('sentiment-analysis')
torch.__version__
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 30
bs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



decompositional_model=   BertForTokenClassification.from_pretrained("model_token_clasiffier")
decompositional_model.to(device)
decompositional_model.eval()


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
df=pd.read_csv("spacy_tag_dep.csv",encoding="utf-8") ### load the concatenation of capacy dependancy and post tags
list_tags=df.Label.values

# gets the cosine similarity between embedings
def similarity_aspect(sentence_rep, label_reps):    return F.cosine_similarity(sentence_rep, label_reps)


def add_spacy_tags_ber_tokenizer_2(tokenizer,list_spacy_tags):
    ####add spacy dependency tag list to bert tokenizer
    for tag in list_spacy_tags:
        tokenizer.add_tokens([tag.lower()])
    return tokenizer


def add_spacy_tags_ber_tokenizer(tokenizer,list_spacy_tags):
    ####add spacy dependency tag and pos tag list to bert tokenizer
    pos_tags=["ADJ","ADP","ADV","AUX","CONJ","CCONJ","DET","INTJ","NOUN","NUM",
              "PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X","SPACE"]
    for tag in list_spacy_tags:
        for pos in pos_tags:
            print(tag.lower()+pos.lower())
            tokenizer.add_tokens([tag.lower()+pos.lower()])
    return tokenizer
####add spacy dependency tag list to bert tokenizer
tokenizer=add_spacy_tags_ber_tokenizer(tokenizer,list_tags)

def pad_sequences_post(seq, maxlen, dtype='long'):
    """Simple post zero-padding function which also trims to maxlen if longer.
    """
    output = []
    for s in seq:
        s=tokenizer.convert_tokens_to_ids(s)
        #print(s)
        if len(s) < maxlen:
            output.append(np.concatenate([s, np.zeros(maxlen - len(s))]))
        else:
            output.append(s[:maxlen])
    return np.array(output, dtype=dtype)

def pad_sequences_post_tag(seq, tag2idx,maxlen, dtype='long'):
    """Simple post zero-padding function which also trims to maxlen if longer.
    """
    output = []
    for s in seq:        
        s2=[tag2idx.get(l) for l in s]
        #print(s2)
        if len(s2) < maxlen:
            output.append(np.concatenate([s2, np.zeros(maxlen - len(s2))]))
        else:
            output.append(s2[:maxlen])
    return np.array(output, dtype=dtype)


# the tag to id for the decompositional model ----- it is based on BIO tagging scheme
tag2idx={'O\n': 0,
         'B-OC\n': 1,
         'I-null\n': 2,
         'B-C\n': 3,
         'B-null\n': 4,
         'I-OC\n': 5,
         'B-OA\n': 6,
         'B-A\n': 7,
         'I-OA\n': 8,
         'I-A\n': 9,
         'I-C\n': 10,
         'PAD': 11}
# 
tag2name={tag2idx[key] : key for key in tag2idx.keys()}

def inference(input_text_dependecy_format,raw_input_text):

    """ uses  BERTForTokenClassifcation model fine-tuned on the three functional compoenents of verbatim 
    
    """
    tag2idx={'O\n': 0,
             'B-OC\n': 1,
             'I-null\n': 2,
             'B-C\n': 3,
             'B-null\n': 4,
             'I-OC\n': 5,
             'B-OA\n': 6,
             'B-A\n': 7,
             'I-OA\n': 8,
             'I-A\n': 9,
             'I-C\n': 10,
             'PAD': 11}

    input_text_dependecy_format=input_text_dependecy_format.lower()
    tokenized_texts = []
    temp_token_dependecy_format = []
    temp_token_raw_text = []
    # Add [CLS] at the front 
    temp_token_dependecy_format.append('[CLS]')
    temp_token_raw_text.append('[CLS]')
    # tokenize dependancy format and the raw input
    token_list_dependecy_format = tokenizer.tokenize(input_text_dependecy_format) 
    token_list_raw_text = tokenizer.tokenize(raw_input_text)
    for m,token in enumerate(token_list_dependecy_format):
        temp_token_dependecy_format.append(token)
    for m,token in enumerate(token_list_raw_text):
        temp_token_raw_text.append(token)

    # Trim the token to fit the length requirement
    if len(temp_token_dependecy_format) > MAX_LEN-1:
        temp_token_dependecy_format= temp_token_dependecy_format[:MAX_LEN-1]
        
    if len(temp_token_raw_text) > MAX_LEN-1:
        temp_token_raw_text= temp_token_raw_text[:MAX_LEN-1]
    # Add [SEP] at the end
    temp_token_dependecy_format.append('[SEP]')
    temp_token_raw_text.append('[SEP]')


    tokenized_texts.append(temp_token_dependecy_format)
    
    
    # pad the tokens to max size
    input_ids=pad_sequences_post(tokenized_texts,maxlen=MAX_LEN, dtype="int64")
    # For fine tune of predict, with token mask is 1,pad token is 0
    attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
    attention_masks[0]

    # identyfy segment ids
    segment_ids = [[0] * len(input_id) for input_id in input_ids]
    segment_ids[0]

    # convert the input to torch tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    segment_ids = torch.tensor(segment_ids)

    # Set save model to Evalue loop and 
    # Get model predict result
    with torch.no_grad():
            outputs = decompositional_model(input_ids.to(device), token_type_ids=None,
            attention_mask=None,)
            # For eval mode, the first result of outputs is logits
            logits = outputs[0]
            # Make logits into numpy type predict result
    # The predict result contain each token's all tags predict result
    predict_results = logits.detach().cpu().numpy()


    result_arrays_soft = softmax(predict_results[0])
    result_array = result_arrays_soft
    # Get each token final predict tag index result
    result_list = np.argmax(result_array,axis=-1)

    # initialize arrays to register the components
    target_concept=[]
    aspect=[]
    target_conceept_opinion=[]
    aspect_opinion=[]
  
    for i, mark in enumerate(attention_masks[0]):
        if mark>0:
            token=temp_token_dependecy_format[i]
            tag=tag2name[result_list[i]]       

            # get aspects
            if (str(tag).strip()=='B-A' or str(tag).strip()=='I-A'): 
                aspect.append(token)
                    
            # get target concepts 
            elif str(tag).strip()=='B-C' or str(tag).strip()=='I-C':             
                target_concept.append(token)                 

            #  gets the opinions
            elif str(tag).strip()=='B-OC' or str(tag).strip()=='I-OC' or str(tag).strip()=='B-OA' or str(tag).strip()=='I-OA':     
                target_conceept_opinion.append(token)              

    return ' '.join(target_concept),' '.join(aspect),' '.join(target_conceept_opinion)





# constract spacy based dependancy stracture for a senetence
def dependecy_tag(senetence):
    tags=[]
    dct_index_tag_token={}
    doc=nlp(senetence)
    for token in doc:
        tags.append(str(token.dep_+token.pos_).lower())
        dct_index_tag_token[token.text]=str(token.dep_+token.pos_).lower()
    return tags,dct_index_tag_token

def pos_tag(senetence):
    tags=[]
    dct_index_tag_token={}
    doc=nlp(senetence)
    for token in doc:
        #tags.append(str(token.dep_+token.pos_).lower())
        dct_index_tag_token[token.text]=token.tag_
    return dct_index_tag_token

def get_components_list(texts_list):
    target_concept,aspects,opinionss,combined,sentiment=[],[],[],[],[]
    texts_all=[]
    splitter = r"[\.?!]"
    for texts in texts_list:   
        #text_all=re.split(splitter, texts) # segement a senence using senetence delimeters      

        text_sentences = nlp(texts) # segments text using spacy
        for sentence in text_sentences.sents:
            text=sentence.text.strip()
            if text!="":
                raw_text=text
                output=sentiment_model(text) # get sentiment category of the sentence
                sent=output[0]['label'] # gets the sentiment category
                score=output[0]['score'] # gets the confidence
                sentiment_cat="NEUTRAL" # make 'neutral' the default sentiment
                if sent in ['NEGATIVE','POSITIVE'] and score>0.95:  
                    sentiment_cat=sent
                tokens,dct_index_tag_token=dependecy_tag(text)              

                dependancy_text=' '.join(tokens) 
                tcs,asps,ops=inference(dependancy_text,raw_text)
                tcs=tcs.replace(" _ ","_").replace("[CLS]","").replace("[SEP]","").replace(" ##","")
                asps=asps.replace(" _ ","_").replace("[CLS]","").replace("[SEP]","").replace(" ##","")
                ops=ops.replace(" _ ","_").replace("[CLS]","").replace("[SEP]","").replace(" ##","")

                tcs_f,asps_f,ops_f=[],[],[] 

                # post process the target concepts          
                tcs_all=tcs.split(" ")
                tcs_n=[]
                for tc in tcs_all:
                    tc=tc.replace(" ","").strip()
                    if tc!="":
                        tcs_n.append(tc)


                for key,value in dct_index_tag_token.items():
                    for tc in tcs_n:
                        if value==tc:
                            if key not in tcs_f:
                                tag_set=['NN','NNS']
                                key=filter_tag(key,tag_set) # consider nouns only
                                tcs_f.append(key)

                # post process the aspects
                asps_all=asps.split(" ")
                asps_n=[]
                for asp in asps_all:
                    asps_n.append(asp.replace(" ",""))

                asps_n_extended=[]
                for asp_n in asps_n:
                    splited_by_space=asp_n.split(" ")
                    for concepts_element in splited_by_space:
                        asps_n_extended.append(concepts_element)

                for key,value in dct_index_tag_token.items():
                    for asp in asps_n_extended:
                        if value==asp:
                            if key not in asps_f:
                                tag_set=['NN','NNS']
                                key=filter_tag(key,tag_set)  # consider nouns only
                                asps_f.append(key)

               # post process the opinions
                ops_all=ops.split(" ")
                ops_n=[]
                for op in ops_all:
                    ops_n.append(op.replace(" ",""))
                for key,value in dct_index_tag_token.items():
                    for op in ops_n:
                        if value==op:
                            if key not in ops_f:
                                ops_f.append(key)
                target_concept.append(' '.join(tcs_f))
                aspects.append(' '.join(asps_f))
                opinionss.append(' '.join(ops_f))
                sentiment.append(sentiment_cat)
                

                            
                texts_all.append(texts) # appen the raw texts
    opinions_description=[]            
    opinion_pos_tag=pos_tag(' '.join(ops_f))
    for token_text,p_tag in opinion_pos_tag.items():
        print(p_tag)
        print(len(target_concept))
        if p_tag in ['NN','NNS','NOUN','NOUNS','NNP']:
            if len(target_concept)==1 and target_concept[0]=="":
                target_concept.append(token_text)
            else:
                aspects.append(token_text)
        else:
            opinions_description.append(token_text) 
            
    return(target_concept,aspects,opinions_description,sentiment,texts_all)
       


def filter_tag(text,tag_set):
    filtered_text=""
    doc = nlp(text)        
    for token in doc:
        if token.tag_ in tag_set and token.tag_ not in ['PRP']:
           filtered_text=filtered_text+" "+token.text
    filtered_text=filtered_text.strip()
    return filtered_text
    
    
    
def filter_tag2(text,tag_set):
    filtered_text=""
    #doc = nlp(text)        
    for token in text:
        #if token.tag_ in tag_set:
        filtered_text=filtered_text+" "+token
    filtered_text=filtered_text.strip()
    return filtered_text
    
    
    
    

def decomposotionality(doc): # return decompsitional elements for p
    sub=[]
    obj=[]
    opinion=[]

    for token in doc:

        if (token.dep_=='nsubj'):
            sub.append(token.text)
        # extract object
        elif (token.dep_=='pobj' or token.dep_=='dobj'):
            #print(token.text)
            obj.append(token.text)
        elif (token.dep_=='amod' or token.dep_=='acomp' or token.dep_=='ROOT'):

            opinion.append(token.text)
    return sub,opinion,obj

def get_componenets_svo(input_text):
    target_concept,aspect,opinion=[],[],[]



    tok1 = nlp(input_text)
    svos1 = findSVOs(tok1)  
    comp1=[list(elem) for elem in svos1]  


    
    for cmp1 in comp1:
        print(len(cmp1))
        if len(cmp1)==3:
            target_concept.append(cmp1[0])
            opinion.append(cmp1[1])
            aspect.append(cmp1[2])
        if len(cmp1)==2:
            target_concept.append(cmp1[0])
            opinion.append(cmp1[1])



            
    return target_concept,aspect,opinion    
    
    
    
