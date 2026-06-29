import os
import pickle
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

with open("chunks.pkl", "rb") as file:
    all_docs = pickle.load(file)

if isinstance(all_docs[0], str):
    texts_to_embed = all_docs
else:
    texts_to_embed = [doc.page_content for doc in all_docs]

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY. Please check your environment variables.")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

collection_name = "agent_knowledge_base"

if not client.collection_exists(collection_name):
    print(f"Creating new cloud collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

total_docs = len(texts_to_embed)
batch_size = 100

for i in range(0, total_docs, batch_size):
    batch_texts = texts_to_embed[i : i + batch_size]
    batch_vectors = embeddings.embed_documents(batch_texts)

    points = []
    for j, (text, vector) in enumerate(zip(batch_texts, batch_vectors)):
        point_id = i + j

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"page_content": text}
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"✅ Uploaded documents {i} to {min(i + batch_size - 1, total_docs - 1)}")
