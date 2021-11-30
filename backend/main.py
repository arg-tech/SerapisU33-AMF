from flask import Flask, render_template, request
#from werkzeug import FileSystemLoader
from jinja2 import Environment, FileSystemLoader
from flask import json
from flask import jsonify
import os
import argparse
import json
import hashlib



from dam3 import get_argument_reletion_dam3

from dam2 import get_inference_dam2
from dam1 import get_inference_dam1
from bert_te import get_inference_bert_te

from segmenter import get_segmenter_default
from propositionalizer import get_propositionalizer_default
from turninator import get_turninator_default



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




app = Flask(__name__)


@app.route('/dam-03', methods = ['GET', 'POST'])
def dam3():
	if request.method == 'POST':
		print("posted")
		f = request.files['file']
		f.save(f.filename)
		ff = open(f.filename,'r')
		content = ff.read()
		result=get_argument_reletion_dam3(f.filename)
	return result

@app.route('/dam-02', methods = ['GET', 'POST'])
def dam2():
	if request.method == 'POST':
		print("posted")
		f = request.files['file']
		f.save(f.filename)
		ff = open(f.filename,'r')
		content = ff.read()
		result=get_inference_dam2(f.filename)
	return result

@app.route('/dam-01', methods = ['GET', 'POST'])
def dam1():
	if request.method == 'POST':
		print("posted")
		f = request.files['file']
		f.save(f.filename)
		ff = open(f.filename,'r')
		content = ff.read()
		result=get_inference_dam1(f.filename)
	return result

@app.route('/bert-te', methods = ['GET', 'POST'])
def bertte():
	if request.method == 'POST':
		print("posted")
		f = request.files['file']
		f.save(f.filename)
		ff = open(f.filename,'r')
		content = ff.read()
		result=get_inference_bert_te(f.filename)
	return result	
	
	
@app.route('/segmenter-01', methods = ['GET', 'POST'])
def segmenter_defult():
	if request.method == 'POST':
		print("posted")
		f = request.files['file']
		f.save(f.filename)
		ff = open(f.filename,'r')
		content = ff.read()
		result=get_segmenter_default(f.filename)
	return result	
#get_propositionalizer_default
#turninator-01, propositionalizer-01,segmenter-01
@app.route('/propositionalizer-01', methods = ['GET', 'POST'])
def propositionalizer_defult():
	if request.method == 'POST':
		print("posted")
		f = request.files['file']
		f.save(f.filename)
		ff = open(f.filename,'r')
		content = ff.read()
		result=get_propositionalizer_default(f.filename)
	return result
	
@app.route('/turninator-01', methods = ['GET', 'POST'])
def turninator_defult():
	if request.method == 'POST':
		print("posted")
		f = request.files['file']
		f.save(f.filename)
		ff = open(f.filename,'r')
		content = ff.read()
		result=get_turninator_default(f.filename)
	return result	
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5009"), debug=False)	  
