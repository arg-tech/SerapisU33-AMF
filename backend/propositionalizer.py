from flask import Flask, render_template, request

from jinja2 import Environment, FileSystemLoader

###################

import os
import pandas as pd
import re
from flask import json
from flask import jsonify

def is_json(myjson):
  try:
    #data=open(myjson)
    #data2=data.read()
    #json_dict = json.loads(data2)
    json_object = json.loads(myjson)
  except ValueError as e:
    print(e)
    return False
  return True



 



def propositionalizer(json_aif):
	is_json_file=is_json(json_aif)
	if is_json_file: 
		json_dict = json.loads(json_aif)
		json_aif={}
		L_nodes=[]
		edges=[]
		locutions=[]
		
		if 'nodes' in json_dict and 'locutions' in json_dict and 'edges' in json_dict:
		
			text_with_span_old=""
			if 'text' in json_dict:
				text_with_span_old=json_dict['text']
			text_with_span="empty"

			nodes=json_dict['nodes']
			nodes_len=len(nodes)
			
			old_locutions=json_dict['locutions']
			old_edges=json_dict['edges']
			j=0
			i=0
			i_nodes_lis=[]
			for nodes_entry in nodes:
				speaker=""
				propositions=nodes_entry['text']
				n_id=nodes_entry['nodeID'] 
				type=nodes_entry['type']
				if propositions not in i_nodes_lis:
					if type=="L": 
						
						for entry_locutions in old_locutions:
							l_id=entry_locutions['nodeID']
							if n_id==l_id:
								speaker=entry_locutions['personID']
								j=j+1
						node_id=j
						L_nodes.append({'text': propositions, 'type':'L','nodeID': node_id})  
						locution_id=node_id+1
						locutions.append({'personID': speaker, 'nodeID': node_id}) 
						inode_id=locution_id+1
						L_nodes.append({'text': propositions, 'type':'I','nodeID': inode_id})
						i_nodes_lis.append(propositions)
						y_id=inode_id+1
						L_nodes.append({'text': 'Default Illocuting', 'type':'YA','nodeID': y_id})	
						edge_id=y_id+1
						edges.append({'toID': y_id, 'fromID':node_id,'edgeID': edge_id})
						edges.append({'toID': inode_id, 'fromID':y_id,'edgeID': edge_id+1})
						i=i+1
						j=edge_id+3               
						   
			json_aif.update( {'nodes' : L_nodes} )
			json_aif.update( {'edges' : edges} )
			json_aif.update( {'locutions' : locutions} )
			
			if text_with_span_old:			

				json_aif.update({'text': text_with_span_old})
			else:
				#print(text_with_span)
				json_aif.update({'text': text_with_span})
			return json_aif
		else:
			return("Incorrect json-aif format")
	else:
		return("Incorrect input format")

#result1=propositionalizer(result) 

####################
def get_propositionalizer_default(notebook_path):
	j=0
	data=open(notebook_path)
	data2=data.read()

	json_aif=propositionalizer(data2) 	

	return json_aif


