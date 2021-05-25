from pymongo import MongoClient

def create_connection(database='agreement',collection='agreement'):
    client = MongoClient("mongodb+srv://Getmeonline1:Getmeonline1@agreement.bfzsl.mongodb.net/agreement?retryWrites=true&w=majority")
    db=client[database]
    collection = db[collection]
    return collection