import pymongo
import pandas as pd




class Database():
    def __init__(self, db_url, db_name, collection_name, port=27017):
        self.db_url = db_url
        self.port = str(port)
        self.db_name = db_name
        self.collection_name = collection_name

        self.client = pymongo.MongoClient(f'mongodb://{self.db_url}:{self.port}')
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]



    def get_latest_data(self, key, limit=1000):
        # Find the youngest document (largest date)
            # Get the latest 10 documents sorted by date_published (descending)
        #latest_data = list(self.collection.find().sort('date_published', -1).limit(limit))
        latest_data = list(self.collection.find().sort(key, -1))

        return latest_data
    
    def get_all_data(self, key):

        data = list(self.collection.find().sort(key, -1))
        return data

    def upload_one(self, data, verbose=False):
        # some database upload logic
        self.collection.insert_one(data)
        if verbose:
            print("Data uploaded successfully!")
    
    def upload_many(self, data:list, verbose=False):
        # some database upload logic
        if data:
            self.collection.insert_many(data)
        if verbose:
            print("Upload Many Data successfully!")

    def get_null_data(self, key):
        null_data = list(self.collection.find({key: None}))
        return null_data
    
    def update_data(self, data):

        # Update document in MongoDB using all updated keys in 'data'
        update_fields = {k: v for k, v in data.items() if k != '_id'}
        self.collection.update_one(
            {"_id": data["_id"]},
            {"$set": update_fields}
        )
        #print(f'Updated data: {data["_id"]}')
        # new_data = self.collection.find_one({"_id": data["_id"]})
        # print(new_data)

        

    def close_connection(self):
        self.client.close()


if __name__ == '__main__':

    db = Database(
        db_url='localhost',
        db_name='deep-learning',
        collection_name='test'
    )

    

    db.close_connection()
        