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
		doc1 = nlp(text1)
		doc2 = nlp(text2)    
		more_elements1=decomposotionality(doc1)
		more_elements2=decomposotionality(doc2)
		tok1 = nlp(text1)
		svos1 = findSVOs(tok1)  
		comp1=[list(elem) for elem in svos1]    
		more_elements1=[list(elem) for elem in more_elements1]    
		comp1=comp1+more_elements1####elements of p1
		tok2 = nlp(text2)
		svos2 = findSVOs(tok2)
		comp2=[list(elem) for elem in svos2]    
		more_elements2=[list(elem) for elem in more_elements2]    
		comp2=comp2+more_elements2### elements of p2


		comp22=[] #remove stop words for p1

		for ll in comp2:# list
			wrd_lst=[]

			for phrase in ll: #pharse
				phrase_wrd=""
				
				tokens=phrase.split()
				for word in tokens:
					if word not in stopwords.words('english'):
						if word:
							print("wordsssssss",word)
							phrase_wrd=phrase_wrd+" "+ word.strip()
				if phrase_wrd:
					wrd_lst.append(phrase_wrd.strip())
			comp22.append(wrd_lst)
			

		comp11=[]  ## remve stop words from p2  
		for ll in comp1:# list
			wrd_lst=[]
			
			for phrase in ll: #pharse
				phrase_wrd=""
				
				tokens=phrase.split()
				for word in tokens:
					if word not in stopwords.words('english'):
						if word:
							print("wordsssssss",word)
							phrase_wrd=phrase_wrd+" "+ word.strip()
				if phrase_wrd:
					wrd_lst.append(phrase_wrd.strip())
			comp11.append(wrd_lst)



		print(comp11," and", comp22)

		print(comp11," and", comp22)		


		entailemt_1=get_entailement(text1,text2)# 
		entailemt_2=get_entailement(text2,text1)



		if(entailemt_1[0]>=entailemt_2[0]):    
			entailemt=entailemt_1
			
		if(entailemt_2[0]>=entailemt_1[0]):    
			entailemt=entailemt_2    



		similarity,antonymy=get_sim(comp11,comp22)    
		if len(antonymy)==0:
			antonymy=[0]
		arg_rel1="none"
		arg_rel1=sim_entail_argrel1(similarity,entailemt,antonymy)


		similarity,antonymy=get_sim(comp22,comp11)
		if len(antonymy)==0:
			antonymy=[0]
		arg_rel2="none"
		arg_rel2=sim_entail_argrel2(similarity,entailemt,antonymy)
		return (final_result(arg_rel2,arg_rel1))

def get_argument_reletion_old(p1p2):
	text1=p1p2[0]
	text2=p1p2[1]
	print(text1,text2)
	
	entailemt=[]
	doc1 = nlp(text1)
	doc2 = nlp(text2)    
	more_elements1=decomposotionality(doc1)
	more_elements2=decomposotionality(doc2)
	tok1 = nlp(text1)
	svos1 = findSVOs(tok1)  
	comp1=[list(elem) for elem in svos1]    
	more_elements1=[list(elem) for elem in more_elements1]    
	comp1=comp1+more_elements1####elements of p1
	tok2 = nlp(text2)
	svos2 = findSVOs(tok2)
	comp2=[list(elem) for elem in svos2]    
	more_elements2=[list(elem) for elem in more_elements2]    
	comp2=comp2+more_elements2### elements of p2


	comp22=[] #remove stop words for p1

	for ll in comp2:# list
		wrd_lst=[]

		for phrase in ll: #pharse
			phrase_wrd=""
			
			tokens=phrase.split()
			for word in tokens:
				if word not in stopwords.words('english'):
					phrase_wrd=phrase_wrd+" "+ word.strip()
			wrd_lst.append(phrase_wrd.strip())
		comp22.append(wrd_lst)
		

	comp11=[]  ## remve stop words from p2  
	for ll in comp1:# list
		wrd_lst=[]
		
		for phrase in ll: #pharse
			phrase_wrd=""
			
			tokens=phrase.split()
			for word in tokens:
				if word not in stopwords.words('english'):
					phrase_wrd=phrase_wrd+" "+ word.strip()
			wrd_lst.append(phrase_wrd.strip())
		comp11.append(wrd_lst)



	print(comp11," and", comp22)


	entailemt_1=get_entailement(text1,text2)# 
	entailemt_2=get_entailement(text2,text1)



	if(entailemt_1[0]>=entailemt_2[0]):    
		entailemt=entailemt_1
		
	if(entailemt_2[0]>=entailemt_1[0]):    
		entailemt=entailemt_2    



	similarity,antonymy=get_sim(comp11,comp22)    
	if len(antonymy)==0:
		antonymy=[0]
	arg_rel1="none"
	arg_rel1=sim_entail_argrel1(similarity,entailemt,antonymy)


	similarity,antonymy=get_sim(comp22,comp11)
	if len(antonymy)==0:
		antonymy=[0]
	arg_rel2="none"
	arg_rel2=sim_entail_argrel2(similarity,entailemt,antonymy)
	return (final_result(arg_rel2,arg_rel1))
           

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




def get_anotnyms(word):
	tag_list=["JJ","JJS","RB","IN","PRP"]
	wrd_tag=pos_tag([word])
	tag=wrd_tag[0][1]
	#print(f'{word} pos tagged: {tag}' )
	#print(filtered_words)
	#RB,JJ,JJS
	antonyms = [] 
	if(tag not in tag_list):

	 
		for syn in wordnet.synsets(word): 
			for l in syn.lemmas(): 

				if l.antonyms(): 
					antonyms.append(l.antonyms()[0].name()) 

	return set(antonyms)

def get_sim(comp1,comp2):
    tag_list=["JJ","JJS","RB","IN","PRP"]
    decompositional_sim=[]
    subj12_anotnym=[]
    three_decompositional_sim=[]
    for cmp1 in comp1:
        #print(cmp1)
        for cm1s in cmp1:
            cm1_split=cm1s.split()
            sim_val=0
            data=[0]
            for cm1 in cm1_split:
				
                for cmp2 in comp2:
					#print(cm1,"versus", cmp2)           


                    for cm2 in cmp2:
                        #print(cm1,cmp2)

                        if(cm2 in list(set(get_anotnyms(cm1)))):
                            subj12_anotnym.append(1)

                        input_tag=[]
                        input_tag.append(cm1)
                        input_tag2=[]
                        input_tag2.append(cm2)
                        
                        #print(input_tag,"and input_tag2",input_tag2)

                        if  cm1:
                            wrd_tag1=pos_tag(input_tag)
                            wrd_tag2=pos_tag(input_tag2)
                            tag1=wrd_tag1[0][1]  
                            tag2=wrd_tag2[0][1] 
                            #print(tag1,"and tag2",tag2)
                            if(tag1 not in tag_list and tag2 not in tag_list):
                                #print("not in", cm1,cmp2)
                                inputs = tokenizer.batch_encode_plus([cm1] + cmp2,return_tensors='pt',pad_to_max_length=True)
                                input_ids = inputs['input_ids']
                                attention_mask = inputs['attention_mask']
                                output = model(input_ids, attention_mask=attention_mask)[0]
                                sentence_rep = output[:1].mean(dim=1)
                                label_reps = output[1:].mean(dim=1)	            
                                similarities = F.cosine_similarity(sentence_rep, label_reps)
                                closest = similarities.argsort(descending=True)         					
            
                                for ind in closest:
                                    data.append(similarities[ind].item())
                                    decompositional_sim.append(similarities[ind].item())
                                sim_val=max(data, key=lambda item: item)  
                                print(sim_val)
                    
                three_decompositional_sim.append(sim_val)
    return(three_decompositional_sim,subj12_anotnym)


def get_sim_old(comp1,comp2):
	tag_list=["JJ","JJS","RB","IN","PRP"]
	decompositional_sim=[]
	subj12_anotnym=[]
	three_decompositional_sim=[]
	for cmp1 in comp1:
		#print(cmp1)
		for cm1s in cmp1:
			cm1_split=cm1s.split()
			sim_val=0
			data=[0]
			for cm1 in cm1_split:
				
				for cmp2 in comp2:
					#print(cm1,"versus", cmp2)           


					for cm2 in cmp2:
						#print(cm1,cmp2)

						if(cm2 in list(set(get_anotnyms(cm1)))):
							subj12_anotnym.append(1)

						input_tag=[]
						input_tag.append(cm1)
						input_tag2=[]
						input_tag2.append(cm2)

						if  cm1:
							wrd_tag1=pos_tag(input_tag)
							tag1=wrd_tag1[0][1]        
							if(tag1 not in tag_list):
								inputs = tokenizer.batch_encode_plus([cm1] + cmp2,return_tensors='pt',pad_to_max_length=True)
								input_ids = inputs['input_ids']
								attention_mask = inputs['attention_mask']
								output = model(input_ids, attention_mask=attention_mask)[0]
								sentence_rep = output[:1].mean(dim=1)
								label_reps = output[1:].mean(dim=1)	            
								similarities = F.cosine_similarity(sentence_rep, label_reps)
								closest = similarities.argsort(descending=True)         					
			
								for ind in closest:
									data.append(similarities[ind].item())
									decompositional_sim.append(similarities[ind].item())
								sim_val=max(data, key=lambda item: item)                 
					
				three_decompositional_sim.append(sim_val)
	return(three_decompositional_sim,subj12_anotnym)




def get_sim_old(comp1,comp2):

	decompositional_sim=[]
	subj12_anotnym=[]
	three_decompositional_sim=[]
	for cmp1 in comp1:
		#print(cmp1)
		for cm1s in cmp1:
			cm1_split=cm1s.split()
			sim_val=0
			data=[0]
			for cm1 in cm1_split:
				
				for cmp2 in comp2:
					#print(cm1,"versus", cmp2)           


					for cm2 in cmp2:
						if(cm2 in list(set(get_anotnyms(cm1)))):
							#print(cm2,"it is anotnym")
							subj12_anotnym.append(1)
					#total_subj12_anotnym.append(max(subj12_anotnym, key=lambda item: item))               
					   
					inputs = tokenizer.batch_encode_plus([cm1] + cmp2,
														 return_tensors='pt',
														 pad_to_max_length=True)
					input_ids = inputs['input_ids']
					attention_mask = inputs['attention_mask']
					output = model(input_ids, attention_mask=attention_mask)[0]
					sentence_rep = output[:1].mean(dim=1)
					label_reps = output[1:].mean(dim=1)	

					
					similarities = F.cosine_similarity(sentence_rep, label_reps)
					closest = similarities.argsort(descending=True)
					

					for ind in closest:
						data.append(similarities[ind].item())
						decompositional_sim.append(similarities[ind].item())
					sim_val=max(data, key=lambda item: item)
			three_decompositional_sim.append(sim_val)
	return(three_decompositional_sim,subj12_anotnym)



def get_entailement(text1,text2):
	total_entailement2=[]

	input_ids = tokenizer_enatilement.encode(text1, text2, return_tensors='pt')
	logits = model_enatilement(input_ids)[0]

	entail_contradiction_logits = logits[:,[0,2]]

	probs = entail_contradiction_logits.softmax(dim=1)
	true_prob = probs[:,1].item() * 100


	total_entailement2.append(true_prob)

	return (total_entailement2)



	###################################first**************
def sim_entail_argrel2(similarity,entailemt,antonymy):
	if sim_feature(similarity) and entailemt[0]>80 and antonymy[0]==0:
		arg_rel2="Inference"
	elif sim_feature(similarity) and entailemt[0]<10:
		arg_rel2="Attack"
	elif antonymy[0]==1 and (entailemt[0]<10 or entailemt[0]>80):
		arg_rel2="Attack"
	else:
		arg_rel2="none"
	return arg_rel2

def sim_entail_argrel1(similarity,entailemt,antonymy):
	print(similarity,entailemt,antonymy)
	if sim_feature(similarity) and entailemt[0]>80:
		arg_rel1="Inference"
	elif sim_feature(similarity) and entailemt[0]<10:
		arg_rel1="Attack"
	elif antonymy[0]==1 and (entailemt[0]<10 or entailemt[0]>80):
		arg_rel1="Attack"
	else:
		arg_rel1="none"
	return arg_rel1

def final_result(arg_rel2,arg_rel1):
	final_result="None"

	if(arg_rel2=="Attack" or arg_rel1=="Attack"):
		#print("the relation is ", "Attacks")
		final_result="CA"
	elif(arg_rel2=="Inference" or arg_rel1=="Inference"):
		#print("the relation is ", "Inference")
		final_result="RA"
	else:
		#print("the relation is ", "none")
		final_result="None"
	return final_result

	#######################################################




def sim_feature(similarity):
	for sim in similarity:
		if sim>0.80:
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


	
def get_inference_dam2(notebook_path):
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
	






  
