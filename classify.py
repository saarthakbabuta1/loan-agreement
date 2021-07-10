from connection import create_connection
import numpy as np,numpy.random
from numpy.core.fromnumeric import size
import requests
from bson.objectid import ObjectId

def classify(paragraph,classes):
    random = list(numpy.random.dirichlet(np.ones(6),size =1))
    j = 0
    for clas in classes:
        classes[clas] = random[0][j]
        j = j+1
    
    return classes

def classify_tags(par,document_id):
    try:
        res_tag1 = []
        res_tag2 = []
        res_tag3 = []
        res_tag4 = []
        res_tag5 = []
        res_tag6 = []
        for i in par:
            res_tag1.append(requests.post("http://127.0.0.1:5000/classify/tag1",json = {"data":i}).json()['data'])
            res_tag2.append(requests.post("http://127.0.0.1:5000/classify/tag2",json = {"data":i}).json()['data'])
            res_tag3.append(requests.post("http://127.0.0.1:5000/classify/tag3",json = {"data":i}).json()['data'])
            res_tag4.append(requests.post("http://127.0.0.1:5000/classify/tag4",json = {"data":i}).json()['data'])
            res_tag5.append(requests.post("http://127.0.0.1:5000/classify/tag5",json = {"data":i}).json()['data'])
            res_tag6.append(requests.post("http://127.0.0.1:5000/classify/tag6",json = {"data":i}).json()['data'])
        tags = {"tag1":res_tag1,"tag2":res_tag2,"tag3":res_tag3,"tag4":res_tag4,"tag5":res_tag5,"tag6":res_tag6}
        db= create_connection()
        for i in tags:
            db.update({'_id': ObjectId('{}'.format(document_id)) },{ "$set" : {i:tags[i]}})
        return "Updated"
    except Exception as e:
        print(e)
        return e
    


