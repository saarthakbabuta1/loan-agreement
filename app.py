import flask
from flask.globals import request
from numpy import frompyfunc
from bson.json_util import dumps
from document import delete_document, upload_document,get_document,get_paragraph_document
import json
import os
from werkzeug.utils import secure_filename
from helper import allowed_file
import time
import pandas as pd
import openpyxl
from classify import random_classification,classify_tags
import requests

app = flask.Flask(__name__)

app.config["DEBUG"] = True

@app.route('/',methods=['GET'])
def home():
    return "Loan Agreements App"

@app.route('/document',methods=['GET','POST','DELETE'])
def document():
    if request.method == 'GET':
        if 'document_id' in flask.request.args:
            document_id = flask.request.args.get('document_id')
            doc = get_document(document_id)
            return {"data":json.loads(dumps(doc))}
        else:
            return {"message":"document id missing"}
    
    if request.method == 'POST':
        start_time = time.time()
        image = flask.request.files['file']
        filename = secure_filename(image.filename)
        if image and allowed_file(filename):
            image.save(filename)
        else:
            return {"message":"File is either null or unsupported"}
        print("File added in the folder")
        upload_document(filename)
        print("Insertion completed in %s seconds" % (time.time() - start_time))
        return {"meassge" : "data inserted successfully"}
        

    if request.method == 'DELETE':
        if 'document_id' in flask.request.args:
            document_id = flask.request.args.get('document_id')
            delete_document(document_id)
            return {"message":"data has been deleted successfuly"}
        else:
            return {"message":"document id missing"}

@app.route('/download',methods=['GET'])
def download():
    
    if 'document_id' in flask.request.args:
        document_id = flask.request.args.get('document_id')
        doc = get_document(document_id)
        doc.pop("data")
        df = pd.DataFrame(data=doc)
        df = df.applymap(lambda x: x.encode('unicode_escape')
        .decode('utf-8') if isinstance(x, str) else x)
        df.to_excel('output.xlsx')
    
    return {"message":"document has been downloaded"}

@app.route('/classify/tags',methods=['POST'])
def classify_tag():
    par = json.loads(request.data)
    tags = random_classification()
    print(tags)
    return tags

@app.route('/classify/<id>',methods=['POST'])
def classify_all(id):
    par = json.loads(dumps(get_paragraph_document(id)))['paragraphs']
    tags = classify_tags(par,id)
    if tags == "Updated":
        return {"message":tags}
    else:
        return {"message":" Not Updated"}


app.run()