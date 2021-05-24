from pymongo import MongoClient

def create_connection():
    client = MongoClient("mongodb+srv://Getmeonline1:Getmeonline1@agreement.bfzsl.mongodb.net/agreement?retryWrites=true&w=majority")
    db=client['agreement']
    collection = db['agreement']
    return collection