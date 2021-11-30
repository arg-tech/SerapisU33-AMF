from flask import Flask, render_template, request
#from werkzeug import FileSystemLoader
from jinja2 import Environment, FileSystemLoader
from flask import json
from flask import jsonify
import os
import argparse
import json
import hashlib


import numpy as np
import re
from tqdm import trange


import numpy as np
import os
from tqdm import trange
from subject_verb_object_extract import findSVOs, printDeps, nlp
import nltk 
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk import pos_tag


from decompose import *
from preprocessors import *

import os
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from transformers import BartForSequenceClassification, BartTokenizer
tokenizer_enatilement = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
model_enatilement = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')




def get_argument_reletion(p1p2):
    text1=p1p2[0]
    text2=p1p2[1]
    print(text1,text2)
  
    entailemt=[]
  		
    d_tc_c,d_asp_c,d_opinion_c,d_sentimnt_c,texts_all_c=get_components_list([text1])# sequence labling based    
  
    svo_tc_c,svo_asp_c,svo_opinion_c=get_componenets_svo(text1)# svo pattern based
    merged_tc_c,merged_asp_c,merged_opinion_c=list(set(svo_tc_c) | set([" ".join(d_tc_c)])),\
        list(set(svo_asp_c) | set([" ".join(d_asp_c)])),\
            list(set(svo_opinion_c) | set([" ".join(d_opinion_c)]))
  		

    d_tc_p,d_asp_p,d_opinion_p,d_sentimnt_p,texts_all_p=get_components_list([text2])# sequence labling based   
  
    svo_tc_p,svo_asp_p,svo_opinion_p=get_componenets_svo(text2)# svo pattern based
    merged_tc_p,merged_asp_p,merged_opinion_p=list(set(svo_tc_p) | set([" ".join(d_tc_p)])),\
        list(set(svo_asp_p) | set([" ".join(d_asp_p)])),\
            list(set(svo_opinion_p) | set([" ".join(d_opinion_p)]))
			
			
	#consider Nouns and VERBS only for tc and aspects
	
    merged_tc_c_clean=[]
    for m_tc_c in merged_tc_c:
        cleaned_tokens=[]
        token_tag_dct=pos_tag(m_tc_c)
        for tok,tag in token_tag_dct.items():
            if tag in ['NNP','NN','VBP','NNS','JJ']:
                cleaned_tokens.append(tok)
        if len(cleaned_tokens)>0:
            merged_tc_c_clean.append(" ".join(cleaned_tokens))
	  
    merged_tc_p_clean=[]
    for m_tc_c in merged_tc_p:
        cleaned_tokens=[]
        token_tag_dct=pos_tag(m_tc_c)
        for tok,tag in token_tag_dct.items():
            if tag in ['NNP','NN','VBP','NNS','JJ']:
                cleaned_tokens.append(tok)
        if len(cleaned_tokens)>0:
            merged_tc_p_clean.append(" ".join(cleaned_tokens))
		
    merged_asp_c_clean=[]
    for m_tc_c in merged_asp_c:
        cleaned_tokens=[]
        token_tag_dct=pos_tag(m_tc_c)
        for tok,tag in token_tag_dct.items():
            if tag in ['NNP','NN','VBP','NNS','JJ']:
                cleaned_tokens.append(tok)			
        if len(cleaned_tokens)>0:
            merged_asp_c_clean.append(" ".join(cleaned_tokens))

    merged_asp_p_clean=[]
    for m_tc_c in merged_asp_p:
        cleaned_tokens=[]
        token_tag_dct=pos_tag(m_tc_c)
        for tok,tag in token_tag_dct.items():
            if tag in ['NNP','NN','VBP','NNS','JJ']:
                cleaned_tokens.append(tok)			
        if len(cleaned_tokens)>0:        
            merged_asp_p_clean.append(" ".join(cleaned_tokens))
        
        
        
        
    # if the target concept, aspect of the premise and conclussion are not recoginized by the models
    # use patterns
    
    if len(merged_tc_c_clean)==0 or len(merged_asp_c_clean)==0:
        doc=nlp(text1)
        c_patterns_tc,c_pattern_asp,c_paterns_ops=decomposotionality(doc)
        
        if len(merged_tc_c_clean)==0:
            merged_tc_c_clean=c_patterns_tc
            
        if len(merged_asp_c_clean)==0:
            merged_asp_c_clean=c_pattern_asp        

    if len(merged_tc_p_clean)==0 or len(merged_asp_p_clean)==0:
        doc=nlp(text2)    
        p_patterns_tc,p_pattern_asp,p_paterns_ops=decomposotionality(doc)
        
        if len(merged_tc_p_clean)==0:
            merged_tc_p_clean=p_patterns_tc
        
        if len(merged_asp_p_clean)==0:
            merged_asp_p_clean=p_pattern_asp            
		
    # similarity of target concept of premise and conclussion
    sim_tc_conclussion_premise=0.0
    list_sim_tc_conclussion_premise=[]		
    for m_tc_c in merged_tc_c_clean:
        for m_tc_p in merged_tc_p_clean:		
            list_sim_tc_conclussion_premise.append(get_sim(m_tc_c,m_tc_p))
        if len(list_sim_tc_conclussion_premise)>0:
  		
            sim_tc_conclussion_premise=max(list_sim_tc_conclussion_premise)
        
  		
    #merged_tc_c,merged_tc_p,merged_asp_p,merged_asp_c
    # similarity of target concept of premise and aspect of conclussion
    sim_tc_conclussion_asp_premise=0.0
    list_sim_tc_conclussion_asp_premise=[]		
    for m_tc_p in merged_tc_p_clean:
        for m_asp_c in merged_asp_c_clean:		
            list_sim_tc_conclussion_asp_premise.append(get_sim(m_tc_p,m_asp_c))		
        if len(list_sim_tc_conclussion_asp_premise)>0:
            sim_tc_conclussion_asp_premise=max(list_sim_tc_conclussion_asp_premise)
        
    # similarity of aspect premise and of target concept  of conclussion  
    sim_asp_conclussion_tc_premise=0.0      
    list_sim_asp_conclussion_tc_premise=[]		
    for m_tc_c in merged_tc_c_clean:
        for m_asp_p in merged_asp_p_clean:		
            list_sim_asp_conclussion_tc_premise.append(get_sim(m_tc_c,m_asp_p))		
        if len(list_sim_asp_conclussion_tc_premise)>0:
            sim_asp_conclussion_tc_premise=max(list_sim_asp_conclussion_tc_premise)
  
    # similarity of aspect of premise and aspect of conclussion
    sim_asp_conclussion_asp_premise=0.0
    list_sim_asp_conclussion_asp_premise=[]		
    for m_asp_p in merged_asp_p_clean:
        for m_asp_c in merged_asp_c_clean:		
            list_sim_asp_conclussion_asp_premise.append(get_sim(m_asp_p,m_asp_c))		
        if len(list_sim_asp_conclussion_asp_premise)>0:
            sim_asp_conclussion_asp_premise=max(list_sim_asp_conclussion_asp_premise)
		

    ############# get anonymy tc_c
          
    antonymys_tc_c=[]			
    for words in merged_tc_c_clean:
        for word in words.split(" "):			
            ants=" ".join(list(get_anotnyms(word)))
            antonymys_tc_c.append(ants)
  			
    ############# get anonymy tc_p
    antonymys_tc_p=[]			
    for words in merged_tc_p_clean:
        for word in words.split(" "):			
            ants=" ".join(list(get_anotnyms(word)))
            antonymys_tc_p.append(ants)
    
    ############# get anonymy asp_c          
    antonymys_asp_c=[]			
    for words in merged_asp_c_clean:
        for word in words.split(" "):			
            ants=" ".join(list(get_anotnyms(word)))
            antonymys_asp_c.append(ants)
  			
    ############# get anonymy asp_p
    antonymys_asp_p=[]			
    for words in merged_asp_p_clean:
        for word in words.split(" "):			
            ants=" ".join(list(get_anotnyms(word)))
            antonymys_asp_p.append(ants)
  		
    entailemt_1=get_entailement(text1,text2)# 
    entailemt_2=get_entailement(text2,text1)
    
    
    if(entailemt_1[0]>=entailemt_2[0]):    
        entailemt=entailemt_1
  			
    if(entailemt_2[0]>=entailemt_1[0]):    
        entailemt=entailemt_2
        
    #entailemt,  antonymys_asp_p,antonymys_asp_c,antonymys_tc_p,antonymys_tc_c  
    #sim_asp_conclussion_asp_premise,sim_tc_conclussion_asp_premise,sim_tc_conclussion_premise
        
    
    arg_rel1= getargument_relation_decomp(entailemt,  
                                antonymys_asp_p,
                                antonymys_asp_c,
                                antonymys_tc_p,
                                antonymys_tc_c,
                                [sim_asp_conclussion_asp_premise],
                                [sim_tc_conclussion_asp_premise],
                                [sim_tc_conclussion_premise],
                                [sim_asp_conclussion_tc_premise],
                                merged_tc_c_clean,
                                merged_tc_p_clean,
                                merged_asp_c_clean,
                                merged_asp_p_clean)
    return (arg_rel1)
def are_anotnyms(string1,string2):
    if len(string1)==0 or len(string2)==0:
        return False
    for word in string1:
        if word in string2:
            return True
    return False
            
def getargument_relation_decomp(entailemt,  
                                antonymys_asp_p,
                                antonymys_asp_c,
                                antonymys_tc_p,
                                antonymys_tc_c,
                                sim_asp_conclussion_asp_premise,
                                sim_tc_conclussion_asp_premise,
                                sim_tc_conclussion_premise,
                                sim_asp_conclussion_tc_premise,
                                merged_tc_c,
                                merged_tc_p,
                                merged_asp_p,
                                merged_asp_c):
    arg_rel2="None"
    
    if sim_feature(sim_tc_conclussion_premise) and entailemt[0]>75 and sim_feature(sim_asp_conclussion_asp_premise):
        arg_rel2="RA"
    elif sim_feature(sim_tc_conclussion_premise) and entailemt[0]<10 and sim_feature(sim_asp_conclussion_asp_premise):
        arg_rel2="CA"
    elif sim_feature(sim_tc_conclussion_premise) and entailemt[0]>75:
        arg_rel2="RA"
    elif sim_feature(sim_asp_conclussion_tc_premise) and entailemt[0]>75:
        arg_rel2="RA"
    elif sim_feature(sim_asp_conclussion_asp_premise) and entailemt[0]<10:
        arg_rel2="CA"
    elif sim_feature(sim_asp_conclussion_asp_premise) and entailemt[0]> 75:
        arg_rel2="RA"
		
    elif sim_feature(sim_asp_conclussion_asp_premise) and entailemt[0]>75:
        arg_rel2="RA"
		
    elif sim_feature(sim_tc_conclussion_asp_premise) and entailemt[0]>75:	
        arg_rel2="RA"
		
    elif sim_feature(sim_tc_conclussion_premise) and entailemt[0]>75 and are_anotnyms(antonymys_asp_c,merged_asp_p):
        arg_rel2="CA"	
		
    elif sim_feature(sim_asp_conclussion_asp_premise) and entailemt[0]>75 and are_anotnyms(antonymys_tc_c,merged_tc_p):
        arg_rel2="CA"			
	#are_anotnyms(string1,string2)
    else:
        arg_rel2="None"
    
    return arg_rel2


def get_anotnyms(word):
	#tag_list=["JJ","JJS","RB","IN","PRP"]
	#wrd_tag=pos_tag([word])
	#tag=wrd_tag[0][1]
	#print(f'{word} pos tagged: {tag}' )
	#print(filtered_words)
	#RB,JJ,JJS
	antonyms = [] 
	#if(tag not in tag_list):

	 
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 

			if l.antonyms(): 
				antonyms.append(l.antonyms()[0].name()) 

	return set(antonyms)

def get_sim(comp1,comp2):
    sim_val=0.00
    if comp1!="" and comp2!="":
        similarity_scores=[]
        inputs = tokenizer.batch_encode_plus([comp1] + [comp2],return_tensors='pt',pad_to_max_length=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        output = model(input_ids, attention_mask=attention_mask)[0]
        sentence_rep = output[:1].mean(dim=1)
        label_reps = output[1:].mean(dim=1)	            
        similarities = F.cosine_similarity(sentence_rep, label_reps)
        closest = similarities.argsort(descending=True)         					

        for ind in closest:
            similarity_scores.append(similarities[ind].item())
            #decompositional_sim.append(similarities[ind].item())
            sim_val=max(similarity_scores, key=lambda item: item)  



    return(sim_val)



def get_entailement(text1,text2):
	total_entailement2=[]

	input_ids = tokenizer_enatilement.encode(text1, text2, return_tensors='pt')
	logits = model_enatilement(input_ids)[0]

	entail_contradiction_logits = logits[:,[0,2]]

	probs = entail_contradiction_logits.softmax(dim=1)
	true_prob = probs[:,1].item() * 100


	total_entailement2.append(true_prob)

	return (total_entailement2)




def decomposotionality(doc): # return decompsitional elements for p
	sub=[]
	obj=[]
	opinion=[]

	for token in doc:

		if (token.dep_=='nsubj' and token.tag_ in ['NNP','NN','VBP','NNS','JJ']):
			sub.append(token.text)
		# extract object
		elif (token.dep_=='pobj' or token.dep_=='dobj') and  token.tag_ in ['NNP','NN','VBP','NNS','JJ']:
			#print(token.text)
			obj.append(token.text)
		elif (token.dep_=='amod' or token.dep_=='acomp' or token.dep_=='ROOT') and  token.tag_ in ['NNP','NN','VBP','NNS','JJ']:

			opinion.append(token.text)
	return sub,opinion,obj


def sim_feature(similarity):
	for sim in similarity:
		print(sim)
		if sim>0.70:
			return True
	return False

def is_json(myjson):
  try:
    data=open(myjson)
    data2=data.read()
    #json_dict = json.loads(data2)
    json_object = json.loads(data2)
  except ValueError as e:
    print(e)
    return False
  return True


	
def get_argument_reletion_dam3(notebook_path):
	is_json_file=is_json(notebook_path)
	if is_json_file:  
		print("hi")
		data=open(notebook_path)
		data2=data.read()
		json_dict = json.loads(data2)
		if 'nodes' in json_dict and 'locutions' in json_dict and 'edges' in json_dict:
			json_aif={}
			L_nodes=[]
			edges=[]
			locutions=[]
			nodes=json_dict['nodes']
			old_locutions=json_dict['locutions']
			old_edges=json_dict['edges']
			text_with_span_old=""
			if 'text' in json_dict:
				text_with_span_old=json_dict['text']
			text_with_span=""
			j=0
			i=0
			count_L_nodes=0
			propositions_all=[]
			propositions_id={}

			for nodes_entry in nodes:
				speaker=""
				n_id=nodes_entry['nodeID']
				type=nodes_entry['type']
				if type=="L":        
					propositions=re.split("[.!?]",nodes_entry['text'])
					for p in propositions:
						p=p.strip()
						
						if p!="":
							

							for entry_locutions in old_locutions:

								l_id=entry_locutions['nodeID']
								if n_id==l_id:
									speaker=entry_locutions['personID']
							j=j+1
							l_node_id=j
							L_nodes.append({'text': p, 'type':'L','nodeID': l_node_id})
							count_L_nodes=count_L_nodes+1
							
							locution_id=l_node_id+1
							
							locutions.append({'personID': speaker, 'nodeID': l_node_id})
							count_L_nodes=count_L_nodes+1
							
							i_id=locution_id+1
							
							L_nodes.append({'text': p, 'type':'I','nodeID': i_id})	
							count_L_nodes=count_L_nodes+1
							y_id=i_id+1
							
							L_nodes.append({'text': 'Default Illocuting', 'type':'YA','nodeID': y_id})	
							count_L_nodes=count_L_nodes+1	
							edge_id=y_id+1
							edges.append({'toID': y_id, 'fromID':l_node_id,'edgeID': edge_id})
							count_L_nodes=count_L_nodes+1
							edge_id=edge_id+1
							edges.append({'toID': i_id, 'fromID':y_id,'edgeID': edge_id})
							count_L_nodes=count_L_nodes+1
							if p not in propositions_all:
								propositions_all.append(p)
								propositions_id.update({p:i_id})
							j=edge_id+1
							text_with_span=text_with_span+" "+speaker+" "+"<span class=\"highlighted\" id=\""+str(l_node_id)+"\">"+p+"</span>.<br><br>"

							


			p1=[]
			p2=[]
			a=0
			b=0
			count_L_nodes=count_L_nodes+200000
			while a <len(propositions_all):
				b=a+1
				while b <len(propositions_all):
					p1.append(propositions_all[a])
					p2.append(propositions_all[b])
					p1p2=[propositions_all[a],propositions_all[b]]
					result=get_argument_reletion(p1p2)
					count_L_nodes=count_L_nodes+1
					
					if result=="RA":
						#
						L_nodes.append({'text': 'Default Inference', 'type':'RA','nodeID': count_L_nodes})	
						edge_id1=count_L_nodes+1
						edge_id2=count_L_nodes+2				
						edges.append({'fromID': propositions_id[propositions_all[a]], 'toID': count_L_nodes,'edgeID':edge_id1})				
						edges.append({'fromID': count_L_nodes, 'toID': propositions_id[propositions_all[b]],'edgeID':edge_id2})
						count_L_nodes=count_L_nodes+2
						
					if result=="CA":				
						#
						L_nodes.append({'text': 'Default Conflict', 'type':'CA','nodeID': count_L_nodes})
						edge_id1=count_L_nodes+1
						edge_id2=count_L_nodes+2				
						edges.append({'fromID': propositions_id[propositions_all[a]], 'toID': count_L_nodes,'edgeID':edge_id1})
						edges.append({'fromID': count_L_nodes, 'toID': propositions_id[propositions_all[b]],'edgeID':edge_id2})
						count_L_nodes=count_L_nodes+2
					count_L_nodes=count_L_nodes+1
					b=b+1
				a=a+1	
			json_aif.update({'nodes':L_nodes} )
			json_aif.update({'edges':edges} )
			json_aif.update({'locutions':locutions} )
			if text_with_span_old:				

				json_aif.update({'text': text_with_span_old})
			else:
				#print(text_with_span)
				json_aif.update({'text': text_with_span})
			return json.dumps(json_aif)
		else:
			return("Invalid json-aif")
	else:
		return("Invalid input")
	
