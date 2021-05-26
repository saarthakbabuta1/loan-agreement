import flask
import cv2
from convert_to_text import get_paragraphs, image_to_text, pdf_to_text
from connection import create_connection
import uuid
import os
from bson.objectid import ObjectId

def upload_document(image):
    if image.rsplit('.', 1)[1].lower() == 'pdf':
        data = pdf_to_text(image)
        par = get_paragraphs(data)
        db = create_connection()
        db.insert_one({"number_of_paragraphs":par[1],
        "paragraphs":par[0],"data":data})
    
        return {"message":"data has been inserted"}
    
    else:
    
        # get the text from image.
        data = image_to_text(cv2.imread(image))
        # get the paragraphs from the image.
        par = get_paragraphs(data)
        db = create_connection()
        # Insert data into the document
        db.insert_one({"number_of_paragraphs":par[1],"paragraphs":par[0],"data":data})

        return {"message":"data has been inserted"}

def get_document(document_id):
    db = create_connection() # create connection
    x = db.find_one({'_id': ObjectId('{}'.format(document_id)) }) # Query from mongodb

    return x

def delete_document(document_id):
    db = create_connection() # Create a connection
    db.delete_one({'_id': ObjectId('{}'.format(document_id))}) # Delete Document
    
    return {"message":"data has been deleted"}




    