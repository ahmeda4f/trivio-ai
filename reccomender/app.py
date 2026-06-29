import os
import time
from datetime import datetime
from bson.objectid import ObjectId
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import redis

app = FastAPI(title="Trivio AI Recommendation Engine")

MONGO_URI = os.environ.get("MONGO_URI")
REDIS_HOST = os.environ.get("REDIS_HOST")

if not MONGO_URI:
    raise ValueError("MONGO_URI is missing from environment variables!")

client = MongoClient(MONGO_URI)
redis_client = redis.from_url(REDIS_HOST, decode_responses=True)

db = client["test"]
user_collection = db["users"]
post_collection = db["posts"]
follow_collection = db["follows"]

def get_posts():
    posts = client['test']['posts']
    return posts

def get_all_posts(user_id=None, seen_posts=None, limit=60):
    posts_collection = get_posts()

    filter_query = {
        "location": "profile"
    }

    if user_id:
        filter_query["authorID"] = {"$ne": ObjectId(user_id)}

    if seen_posts:
        seen_object_ids = [ObjectId(pid) for pid in seen_posts if pid]
        if seen_object_ids:
            filter_query["_id"] = {"$nin": seen_object_ids}

    pipeline = [
        {
            "$match": filter_query
        },
        {
            "$limit": limit
        },
        {
            "$lookup": {
                "from": "users",
                "localField": "authorID",
                "foreignField": "_id",
                "as": "author_details"
            }
        },
        {
            "$unwind": {
                "path": "$author_details",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$set": {
                "authorID": "$author_details"
            }
        },
        {
            "$unset": "author_details"
        },
        {
            "$lookup": {
                "from": "posts",
                "localField": "sharedFrom",
                "foreignField": "_id",
                "as": "shared_post"
            }
        },
        {
            "$unwind": {
                "path": "$shared_post",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$lookup": {
                "from": "users",
                "localField": "shared_post.authorID",
                "foreignField": "_id",
                "as": "shared_author"
            }
        },
        {
            "$unwind": {
                "path": "$shared_author",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$set": {
                "sharedFrom": {
                    "$cond": {
                        "if": {"$ifNull": ["$shared_post._id", False]},
                        "then": {
                            "$mergeObjects": [
                                "$shared_post",
                                {"authorID": "$shared_author"}
                            ]
                        },
                        "else": "$sharedFrom"
                    }
                }
            }
        },
        {
            "$unset": ["shared_post", "shared_author"]
        }
    ]

    return list(posts_collection.aggregate(pipeline))    

def get_seen_posts(userId):
    return redis_client.smembers(f"user:{userId}:seen")

def serialize_mongo_doc(data):
    if isinstance(data, list):
        return [serialize_mongo_doc(item) for item in data]
    elif isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key in ["post_embedding", "user_profile_vector"]:
                continue
            new_dict[key] = serialize_mongo_doc(value)
        return new_dict
    elif isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data

def get_ai_feed(user_id, seen_posts, limit=60):
    user = user_collection.find_one(
        {"_id": ObjectId(user_id)},
        {"user_profile_vector": 1}
    )

    if not user:
        return []

    user_vector = user.get("user_profile_vector", [])

    if not user_vector or len(user_vector) != 384:
        return []

    seen_object_ids = [ObjectId(pid) for pid in seen_posts if pid]

    filter_query = {
        "location": "profile",
        "authorID": {"$ne": ObjectId(user_id)}
    }

    if seen_object_ids:
        filter_query["_id"] = {"$nin": seen_object_ids}

    vector_search = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "post_embedding",
            "queryVector": user_vector,
            "numCandidates": 100,
            "limit": limit,
            "filter": filter_query
        }
    }

    pipeline = [
        vector_search,
        {
            "$lookup": {
                "from": "users",
                "localField": "authorID",
                "foreignField": "_id",
                "as": "populated_author"
            }
        },
        {
            "$unwind": {
                "path": "$populated_author",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$set": {"authorID": "$populated_author"}
        },
        {
            "$unset": "populated_author"
        },
        {
            "$lookup": {
                "from": "posts",
                "localField": "sharedFrom",
                "foreignField": "_id",
                "as": "populated_shared"
            }
        },
        {
            "$unwind": {
                "path": "$populated_shared",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$lookup": {
                "from": "users",
                "localField": "populated_shared.authorID",
                "foreignField": "_id",
                "as": "shared_author"
            }
        },
        {
            "$unwind": {
                "path": "$shared_author",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$set": {
                "sharedFrom": {
                    "$cond": {
                        "if": {"$ifNull": ["$populated_shared._id", False]},
                        "then": {
                            "$mergeObjects": [
                                "$populated_shared",
                                {"authorID": "$shared_author"}
                            ]
                        },
                        "else": "$sharedFrom"
                    }
                }
            }
        },
        {
            "$unset": ["populated_shared", "shared_author"]
        }
    ]

    return list(post_collection.aggregate(pipeline))

def get_user_feed(user_id, posts, gravity=1.5, top_k=30):
    current_time = time.time()
    user_id_obj = ObjectId(user_id) if isinstance(user_id, str) else user_id

    user_data = user_collection.find_one(
        {"_id": user_id_obj},
        {"favEntities": 1}
    )

    user_tags = user_data.get("favEntities", []) if user_data else []

    cursor_follow = follow_collection.find(
        {"followerId": user_id_obj},
        {"userId": 1}
    )

    following_ids = {
        doc["userId"] for doc in cursor_follow if "userId" in doc
    }

    for post in posts:
        base = 1.0
        post_tags = post.get("tags_id", [])

        if any(tag in user_tags for tag in post_tags):
            base += 7.0

        author = post.get("authorID")
        author_id = author.get("_id") if isinstance(author, dict) else author

        if author_id in following_ids:
            base += 4.0

        reactions = post.get("reactionCounts", {})
        total_reactions = sum(reactions.values()) if isinstance(reactions, dict) else 0
        comments_count = post.get("commentsCount", 0)

        engagement = (total_reactions * 0.6) + (comments_count * 1.2)
        base += engagement / 100.0

        post_time = post.get("createdAt")

        if isinstance(post_time, datetime):
            post_time = post_time.timestamp()
        elif not post_time:
            post_time = current_time

        hours_old = (current_time - post_time) / 3600.0
        hours_old = max(0, hours_old)

        score = base / ((hours_old + 1.5) ** gravity)
        post["feed_score"] = score

    sorted_posts = sorted(
        posts,
        key=lambda p: p.get("feed_score", 0),
        reverse=True
    )

    return sorted_posts[:top_k]

@app.get("/recommend/{user_id}")
def recommend_posts_endpoint(user_id: str):
    try:
        seen_posts = get_seen_posts(user_id)
        ai_feed = get_ai_feed(user_id, seen_posts)
        if len(ai_feed) == 0:
            posts = get_all_posts(user_id=user_id, seen_posts=seen_posts)
        else:
            posts = ai_feed
            
        final_feed = get_user_feed(user_id=user_id, posts=posts)
        clean_feed = [serialize_mongo_doc(post) for post in final_feed]

        return {
            "status": "success",
            "data": clean_feed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))