from connection import create_connection
import numpy as np,numpy.random
from numpy.core.fromnumeric import size
import requests
from bson.objectid import ObjectId
from tag_classes import classifications

import random


def random_classification():
    random_classifcations = {}
    for tag in classifications.keys():
        random_classifcations[tag] = random.choice(classifications[tag])
    return random_classifcations

def classify_tags(par,document_id):
    try:
        tag_1_applicability = []
        tag_2_area = []
        tag_3_covenant_type = []
        tag_4_covenant_title_tag = []
        tag_5_covenant_description_sub_tags = []
        Tag_6_User_Defined = []
        for i in par:
            res = requests.post("http://127.0.0.1:5000/classify/tags",json = {"data":i}).json()
 
            tag_1_applicability.append(res["tag_1_applicability"])
            tag_2_area.append(res["tag_2_area"])
            tag_3_covenant_type.append(res["tag_3_covenant_type"])
            tag_4_covenant_title_tag.append(res["tag_4_covenant_title_tag"])
            tag_5_covenant_description_sub_tags.append(res["tag_5_covenant_description_sub_tags"])
            Tag_6_User_Defined.append(res["Tag_6_User_Defined"])

        tags = {"tag_1_applicability":tag_1_applicability,"tag_2_area":tag_2_area,
        "tag_3_covenant_type":tag_3_covenant_type,"tag_4_covenant_title_tag":tag_4_covenant_title_tag,
        "tag_5_covenant_description_sub_tags":tag_5_covenant_description_sub_tags,
        "Tag_6_User_Defined":Tag_6_User_Defined}

        db= create_connection()
        for i in tags:
            db.update({'_id': ObjectId('{}'.format(document_id)) },{ "$set" : {i:tags[i]}})
        print("Tags inserted in Document ID {}".format(document_id))    
        return "Updated"
    except Exception as e:
        print(e)
        return e
    


