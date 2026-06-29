import pymongo
import json
import re
import unicodedata
import os
from datetime import datetime

MONGO_URI = os.environ.get("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client['test']
user_collection = db['users']
post_collection = db['posts']

data_path = "trivio_synonyms.json" 
data = json.load(open(data_path, "r"))
mapping = {}

def normalize(text):
    if not isinstance(text, str): text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r'[^a-zA-Z0-9\u0621-\u064A\s]', ' ', text)
    return re.sub(r"\s+", " ", text).strip().lower()

for id_key, synonyms in data.items():
    for name in synonyms:
        mapping[normalize(name)] = id_key

def search_entity(query):
    query = normalize(query)
    if query in mapping:
        player_id = mapping[query]
        return {"id": player_id, "primary_name": data[player_id][0]}
    return {"match_found": False}

def update_users_and_posts():
    print(f"[{datetime.now()}] Starting tag update job...")
    users = user_collection.find()     
    for user in users:
        tags_set = set()
        interests = user.get("favTeams", []) + user.get("favPlayers", [])
        
        for item in interests:
            entity = search_entity(item)
            if entity.get("id"):
                tags_set.add(entity["id"])
                
        if tags_set:
            user_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"favEntities": list(tags_set)}}
            )

    posts = post_collection.find({"tags_processed": {"$ne": True}})
    for post in posts:
        tags_set = set()
        
        for tag in post.get("tags", []):
            entity = search_entity(tag)
            if entity.get("id"):
                tags_set.add(entity["id"])

        caption = normalize(post.get("caption", ""))
        for word in caption.split():
            entity = search_entity(word)
            if entity.get("id"):
                tags_set.add(entity["id"])

        post_collection.update_one(
            {"_id": post["_id"]},
            {"$set": {
                "tags_id": list(tags_set),
                "tags_processed": True  
            }}
        )
        
    print(f"[{datetime.now()}] Job completed successfully.")

if __name__ == "__main__":
    update_users_and_posts()
