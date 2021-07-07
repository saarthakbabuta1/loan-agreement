from typing import Text
import flask
from numpy import imag
import cv2
from bounding_region_text import *
from connection import create_connection
import uuid
import os
from bson.objectid import ObjectId
from classify import classify,tags

def upload_document(image):
    try:
        if image.rsplit('.', 1)[1].lower() == 'pdf':
            try:
                text = pdf_to_text(image)
                print("PDF has been converted to text")
                par_heading = get_paragraphs(text["paragraph"])
                par = par_heading["paragraph"]
                heading = par_heading["heading"]
                pagragraph_classification = classify(par)
                paragraph_tags = tags(par)
                db = create_connection()
                print(db)                
                db.insert_one({"number_of_paragraphs":len(par),
                "paragraph_classification":pagragraph_classification,
                    "paragraphs":par,"heading":heading,"data":text["body"],"tag1":paragraph_tags["tag1"]})
                print("data inserted")
                
                return {"message":"PDF documnet has been inserted"}
            except Exception as e:
                return(e)

        elif image.rsplit('.', 1)[1].lower() in ['docx','doc']:
            try:
                par = word_to_text(image)
                pagragraph_classification = classify(par[0])
                db = create_connection()
                db.insert_one({"number_of_paragraphs":par[1],"paragraph_classification":pagragraph_classification,
                "paragraphs":par[0]})

                return {"message":"Word data has been inserted"}
            except Exception as e:
                return(e)            
        else:
            try:            
                # get the text from image.
                data = image_to_text(image)["body"]
                # get the paragraphs from the image.
                par = get_paragraphs(image_to_text(image)["paragraph"])
                db = create_connection()
                # Insert data into the document
                db.insert_one({"number_of_paragraphs":len(par),"paragraphs":par,"data":data})

                return {"message":"Image data has been inserted"}
            except Exception as e:
                return(e)
    except Exception as e:
        return(e)

def get_document(document_id):
    try:
        db = create_connection() # create connection
        x = db.find_one({'_id': ObjectId('{}'.format(document_id)) }) # Query from mongodb

        return x
    except Exception as e:
        return(e)

def delete_document(document_id):
    try:
        db = create_connection() # Create a connection
        db.delete_one({'_id': ObjectId('{}'.format(document_id))}) # Delete Document
        
        return {"message":"data has been deleted"}
    except Exception as e:
        return(e)




    