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



import os
from torch.nn import functional as F
from transformers import BartForSequenceClassification, BartTokenizer
tokenizer_enatilement = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
model_enatilement = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')










def get_argument_reletion(p1p2):
	text1=p1p2[0]
	text2=p1p2[1]

	return (get_entailement(text1,text2))
           
def get_entailement(text1,text2):
    arg_rel2="None"
    total_entailement2=[]
    true_prob=0.0
    

    input_ids = tokenizer_enatilement.encode(text1, text2, return_tensors='pt')
    logits = model_enatilement(input_ids)[0]

    entail_contradiction_logits = logits[:,[0,2]]

    probs = entail_contradiction_logits.softmax(dim=1)
    #print("prob 1",probs)
    true_prob1 = probs[:,1].item() * 100
    
    print(true_prob1)
    
    
    
    input_ids = tokenizer_enatilement.encode(text2, text1, return_tensors='pt')
    logits = model_enatilement(input_ids)[0]

    entail_contradiction_logits = logits[:,[0,2]]

    probs = entail_contradiction_logits.softmax(dim=1)
    #print("prob 2",probs)
    true_prob2 = probs[:,1].item() * 100
    
    print(true_prob2)

    if(true_prob2>=true_prob1):
        true_prob=true_prob2
    else:
        true_prob=true_prob1
        
    total_entailement2.append(true_prob)

    if  total_entailement2[0]>80:
        arg_rel2="RA"
    elif  total_entailement2[0]<10:
        arg_rel2="CA"
    else:
        arg_rel2="None"

    return (arg_rel2) 

def get_entailement_old(text1,text2):
    arg_rel2="None"
    total_entailement2=[]

    input_ids = tokenizer_enatilement.encode(text1, text2, return_tensors='pt')
    logits = model_enatilement(input_ids)[0]

    entail_contradiction_logits = logits[:,[0,2]]

    probs = entail_contradiction_logits.softmax(dim=1)
    true_prob = probs[:,1].item() * 100


    total_entailement2.append(true_prob)

    if  total_entailement2[0]>80:
        arg_rel2="RA"
    elif  total_entailement2[0]<10:
        arg_rel2="CA"
    else:
        arg_rel2="None"

    return (arg_rel2) 




	###################################first**************





	###################################first**************
	






	
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


	
def get_inference_bert_te(notebook_path):
	is_json_file=is_json(notebook_path)
	if is_json_file: 
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
							


			p1=[]
			p2=[]
			a=0
			b=0
			while a <len(propositions_all):
				b=a+1
				while b <len(propositions_all):
					p1.append(propositions_all[a])
					p2.append(propositions_all[b])
					p1p2=[propositions_all[a],propositions_all[b]]
					result=get_argument_reletion(p1p2)
					count_L_nodes=count_L_nodes+2000
					if result=="RA":
						count_L_nodes=count_L_nodes+2
						L_nodes.append({'text': 'Default Inference', 'type':'RA','nodeID': count_L_nodes})					
						edges.append({'fromID': propositions_id[propositions_all[a]], 'toID': count_L_nodes,'edgeID ':count_L_nodes+3})				
						edges.append({'fromID': count_L_nodes, 'toID': propositions_id[propositions_all[b]],'edgeID ':count_L_nodes+4})
						count_L_nodes=count_L_nodes+1
					if result=="CA":				
						count_L_nodes=count_L_nodes+2
						L_nodes.append({'text': 'Default Conflict', 'type':'CA','nodeID': count_L_nodes})				
						edges.append({'fromID': propositions_id[propositions_all[a]], 'toID': count_L_nodes,'edgeID ':count_L_nodes+3})
						edges.append({'fromID': count_L_nodes, 'toID': propositions_id[propositions_all[b]],'edgeID ':count_L_nodes+4})
						count_L_nodes=count_L_nodes+1
					count_L_nodes=count_L_nodes+1
					b=b+1
				a=a+1	
			json_aif.update({'nodes':L_nodes} )
			json_aif.update({'edges':edges} )
			json_aif.update({'locutions':locutions} )
			#return json.dumps(json_aif)
			
			if text_with_span_old:	
				json_aif.update({'text': text_with_span_old})
			else:
				json_aif.update({'text': text_with_span})
			return json.dumps(json_aif)
		else:
			return("Invalid json-aif")
	else:
		return("Invalid input")





  
