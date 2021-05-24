import flask
from flask.globals import request
from bson.json_util import dumps
from document import delete_document, upload_document,get_document
import json

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
        if 'image' in flask.request.args:
            image = flask.request.args.get('image')
            upload_document(image)
            return {"meassge" : "data inserted successfully"}
        else:
            return {"message":"image is missing"}

    if request.method == 'DELETE':
        document_id = flask.request.args.get('document_id')
        delete_document(document_id)
        return {"message":"data has been deleted successfuly"}

app.run()