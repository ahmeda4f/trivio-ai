import os
import pymongo
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer

MONGO_URI = os.environ.get("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client['test']
user_collection = db['users']
post_collection = db['posts']
reaction_collection = db['reactions']
comment_collection = db['comments'] 

print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def process_post_embeddings():
    """Generates vectors for new posts."""
    print("\n--- STARTING POST EMBEDDINGS ---")
    posts = list(post_collection.find({"post_embedding": {"$exists": False}}))
    print(f"Found {len(posts)} new posts to embed.")

    for post in posts:
        text_to_embed = []
        if post.get('caption'):
            text_to_embed.append(post['caption'])
        if post.get('tags') and len(post['tags']) > 0:
            text_to_embed.extend(post['tags'])

        if text_to_embed:
            combined_text = " ".join(text_to_embed)
            post_embedding = model.encode(combined_text)
            
            post_collection.update_one(
                {"_id": post["_id"]},
                {"$set": {"post_embedding": post_embedding.tolist()}}
            )
            print(f"Embedded Post: {post['_id']}")


def process_user_embeddings():
    """Calculates the average vector for users based on likes, comments, and favorites."""
    print("\n--- STARTING USER EMBEDDINGS ---")
    users = list(user_collection.find())

    for user in users:
        liked_cursor = reaction_collection.find({"userId": user['_id'], "onModel": "post"})
        liked_post_ids = [like['modelId'] for like in liked_cursor]
        
        commented_cursor = comment_collection.find({"userId": user['_id']})
        commented_post_ids = [comment['postId'] for comment in commented_cursor]
        
        interacted_post_ids = liked_post_ids + commented_post_ids
        
        user_vector_embedding = None
        count = 0

        for post_id in interacted_post_ids:
            post = post_collection.find_one({"_id": ObjectId(post_id)})
            
            if post and 'post_embedding' in post:
                embedding = post['post_embedding']
                if user_vector_embedding is None:
                    user_vector_embedding = [0.0] * len(embedding)
                
                user_vector_embedding = [sum(x) for x in zip(user_vector_embedding, embedding)]
                count += 1

        if user_vector_embedding and count > 0:
            user_vector_embedding = [val / count for val in user_vector_embedding]
        
        if user_vector_embedding is None:
            all_favorites = []        
            if user.get('favTeams'):
                all_favorites.extend(user['favTeams'])
            if user.get('favPlayers'):
                all_favorites.extend(user['favPlayers'])

            if len(all_favorites) > 0:
                combined_text = " ".join([str(entity) for entity in all_favorites])
                user_vector_embedding = model.encode(combined_text).tolist()
                
        if user_vector_embedding is not None:
            user_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"user_profile_vector": user_vector_embedding}}
            )
            print(f"Updated Vector for user {user['_id']}")
        else:
            print(f"Skipped user {user['_id']} - no data.")


def process_new_likes_batch():
    print("Checking for new likes to update user vectors...")
    
    new_likes = list(reaction_collection.find({
        "onModel": "post", 
        "ai_processed": {"$ne": True} 
    }))

    if not new_likes:
        print("No new likes to process.")
        return

    print(f"Found {len(new_likes)} new likes.")

    for like in new_likes:
        success = update_user_vector_on_interaction(like['userId'], like['modelId'])
        
        if success:
            reaction_collection.update_one(
                {"_id": like["_id"]},
                {"$set": {"ai_processed": True}}
            )

    print("Likes batch processing complete.")


def process_new_comments_batch():
    print("Checking for new comments to update user vectors...")
    
    new_comments = list(comment_collection.find({
        "ai_processed": {"$ne": True} 
    }))

    if not new_comments:
        print("No new comments to process.")
        return

    print(f"Found {len(new_comments)} new comments.")

    for comment in new_comments:
        success = update_user_vector_on_interaction(comment['userId'], comment['postId'])
        
        if success:
            comment_collection.update_one(
                {"_id": comment["_id"]},
                {"$set": {"ai_processed": True}}
            )

    print("Comments batch processing complete.")


def update_user_vector_on_interaction(user_id, post_id):
    """
    Shifts the user's preference vector in real-time when they interact (like/comment).
    Formula: (Old_User_Vector * 0.85) + (New_Post_Vector * 0.15)
    """
    user_id = ObjectId(user_id)
    post_id = ObjectId(post_id)

    user = user_collection.find_one({"_id": user_id}, {"user_profile_vector": 1})
    post = post_collection.find_one({"_id": post_id}, {"post_embedding": 1})

    if not post or "post_embedding" not in post:
        print(f"Cannot update vector: Post {post_id} lacks an embedding.")
        return False

    post_vector = post["post_embedding"]
    old_user_vector = user.get("user_profile_vector", []) if user else []

    if not old_user_vector or len(old_user_vector) != 384:
        new_user_vector = post_vector 
    else:
        new_user_vector = [
            (old_val * 0.85) + (post_val * 0.15) 
            for old_val, post_val in zip(old_user_vector, post_vector)
        ]

    user_collection.update_one(
        {"_id": user_id},
        {"$set": {"user_profile_vector": new_user_vector}}
    )
    return True

if __name__ == "__main__":
    process_post_embeddings()
    process_user_embeddings()
    process_new_likes_batch()
    process_new_comments_batch()
    print("\n✅ AI Embedding process completed successfully!")
