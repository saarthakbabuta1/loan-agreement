from pymongo import MongoClient

def create_connection(database='agreement',collection='agreement'):
    try:
        client = MongoClient("mongodb+srv://Getmeonline1:Getmeonline1@agreement.bfzsl.mongodb.net/agreement?retryWrites=true&w=majority")
        db=client['agreement']
        collection = db['agreement']
        return collection
    except Exception as e:
        print(e)
        return(e)