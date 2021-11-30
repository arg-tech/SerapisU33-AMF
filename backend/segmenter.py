from flask import Flask, render_template, request
from jinja2 import Environment, FileSystemLoader

###################

import os

import re
from flask import json
from flask import jsonify



####################

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


	
def get_segmenter_default(notebook_path):
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

							


			json_aif.update({'nodes':L_nodes} )
			json_aif.update({'edges':edges} )
			json_aif.update({'locutions':locutions} )

			json_aif.update({'text': text_with_span})
			return json.dumps(json_aif)
		else:
			return("Invalid json-aif")
	else:
		return("Invalid input")
	






  
