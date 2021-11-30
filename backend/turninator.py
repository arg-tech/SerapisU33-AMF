from flask import Flask, render_template, request

from jinja2 import Environment, FileSystemLoader

###################

import os
import pandas as pd
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



 
      

def get_turninator_default(path):
	if path.endswith("json"):
	
		is_json_file=is_json(path)
		json_aif={}
		L_nodes=[]
		edges=[]
		locutions=[]
		if is_json_file: 
			data=open(path)
			data2=data.read()		
			json_dict = json.loads(data2)
			if 'nodes' in json_dict and 'locutions' in json_dict and 'edges' in json_dict:
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
				for nodes_entry in nodes:
					speaker=""
					n_id=nodes_entry['nodeID']
					type=nodes_entry['type']
					if type=="L":        

						l_node_text=nodes_entry['text']
						l_node_text=l_node_text+"\n"
						propositions = re.findall(r'(\w+:)(.*\n)', l_node_text)
						for a in propositions:
							speaker=a[0]
							text=a[1].replace("\n","")
							L_nodes.append({'text': text, 'type':'L','nodeID': j})
							locutions.append({'personID': speaker, 'nodeID': j})
							text_with_span=text_with_span+" "+speaker+" "+"<span class=\"highlighted\" id=\""+str(j)+"\">"+text+"</span>.<br><br>"
							j=j+1
				json_aif.update( {'nodes' : L_nodes} )
				json_aif.update( {'edges' : edges} )
				json_aif.update( {'locutions' : locutions} )
				return json.dumps(json_aif)
			else:
				return("Invalid json-aif")
		else:
			return("Invalid json")

		#return json.dumps(json_aif)
	
	else:
		j=0
		data=open(path)
		data2=data.read()
		data2=data2+"\n"
		json_aif={}
		text_with_span=""
		L_nodes=[]
		edges=[]
		locutions=[]
		propositions = re.findall(r'(\w+:)(.*\n)', data2)
		for a in propositions:
			speaker=a[0]
			text=a[1].replace("\n","")
			L_nodes.append({'text': text, 'type':'L','nodeID': j})
			locutions.append({'personID': speaker, 'nodeID': j})
			text_with_span=text_with_span+" "+speaker+" "+"<span class=\"highlighted\" id=\""+str(j)+"\">"+text+"</span>.<br><br>"
			j=j+1
		json_aif.update( {'nodes' : L_nodes} )
		json_aif.update( {'edges' : edges} )
		json_aif.update( {'locutions' : locutions} )
		

		return json.dumps(json_aif)

  
