import pickle
from fastapi.responses import JSONResponse
from fastapi.responses import ORJSONResponse
import json
import numpy as np
import cv2
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from insightface.app import FaceAnalysis

# ==============================
# CONFIG
# ==============================
THRESHOLD = 0.5   
MODEL_NAME = "buffalo_l"

app = FastAPI()

print("🔄 Loading InsightFace model...")
face_app = FaceAnalysis(name=MODEL_NAME)
face_app.prepare(ctx_id=-1)  

print("📦 Loading Database...")
with open("db.pkl", "rb") as f:
    db_loaded = pickle.load(f)

with open("players_meta.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)

# ==============================
# PREPARE EMBEDDINGS
# ==============================
known_names = []
known_embeddings = []

for name, embs in db_loaded.items():
    for emb in embs:
        known_names.append(name)
        known_embeddings.append(emb)

known_embeddings = np.array(known_embeddings).astype("float32")

known_embeddings = known_embeddings / np.linalg.norm(
    known_embeddings, axis=1, keepdims=True
)

dimension = known_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(known_embeddings)

print("✅ Resources Loaded.")

# ==============================
# ROUTES
# ==============================
@app.get("/")
def home():
    return {"status": "InsightFace Face ID Service Running 🚀"}


@app.post("/recognize",response_class=ORJSONResponse)
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    identified_players = []

    try:
        faces = face_app.get(img)
    except Exception as e:
        return {"players": [], "error": str(e)}

    if len(faces) == 0:
        return {"players": [], "message": "No faces detected"}

    for face in faces:
        query_emb = face.embedding.astype("float32")

        norm = np.linalg.norm(query_emb)
        if norm != 0:
            query_emb = query_emb / norm

        D, I = index.search(query_emb.reshape(1, -1), k=1)
        score = D[0][0]

        if score > THRESHOLD:
            idx = I[0][0]
            folder_name = known_names[idx]
            # print(folder_name)
            player_info = mapping.get(folder_name, {"name": folder_name})
            # print(player_info)

            if player_info not in identified_players:
                identified_players.append(player_info)

    return {"players":identified_players}