from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import time
from sentence_transformers import SentenceTransformer


app = FastAPI(title="Semantic Search with Re-ranking")

# Load models once
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading re-ranker model...")


print("Models loaded!")

class SearchRequest(BaseModel):
    query: str
    k: int = 7
    rerank: bool = True
    rerankK: int = 4


# Dummy 60 support ticket documents
documents = [
    {"id": 0, "content": "User unable to login after password reset.", "metadata": {"source": "support"}},
    {"id": 1, "content": "Payment failed during subscription renewal.", "metadata": {"source": "support"}},
    {"id": 2, "content": "System experiencing slow performance during peak hours.", "metadata": {"source": "support"}},
    {"id": 3, "content": "Customer reports cloud scalability issues when traffic increases.", "metadata": {"source": "support"}},
    {"id": 4, "content": "Error while integrating third-party API service.", "metadata": {"source": "support"}},
    {"id": 5, "content": "User account suspended without notification email.", "metadata": {"source": "support"}},
    {"id": 6, "content": "Billing invoice shows incorrect tax calculation.", "metadata": {"source": "support"}},
    {"id": 7, "content": "Password reset link expired immediately.", "metadata": {"source": "support"}},
    {"id": 8, "content": "Application crashes when uploading large files.", "metadata": {"source": "support"}},
    {"id": 9, "content": "Database connection timeout errors observed.", "metadata": {"source": "support"}},
]

# Expand to 60 by adding variations
base_topics = [
    "login failure issue",
    "payment processing error",
    "cloud scalability problem",
    "slow system performance",
    "password reset malfunction",
    "email notification not received",
    "account suspension dispute",
    "API integration failure",
    "billing calculation mistake",
    "data synchronization error"
]

for i in range(10, 60):
    topic = base_topics[i % len(base_topics)]
    documents.append({
        "id": i,
        "content": f"Support ticket regarding {topic} reported by enterprise customer.",
        "metadata": {"source": "support"}
    })

doc_texts = [doc["content"] for doc in documents]

# Precompute embeddings once
doc_embeddings = embed_model.encode(doc_texts, normalize_embeddings=True)


def retrieve(query, k):
    query_emb = embed_model.encode(query, normalize_embeddings=True)
    scores = np.dot(doc_embeddings, query_emb)

    top_indices = np.argsort(scores)[::-1][:k]
    return [(documents[i], float(scores[i])) for i in top_indices]


def rerank_results(query, candidates, rerankK):
    # Slightly boost semantic similarity scores
    boosted = []

    for doc, score in candidates:
        # small keyword boost
        if query.lower() in doc["content"].lower():
            score += 0.05

        boosted.append((doc, score))

    # Normalize 0-1
    scores = np.array([score for _, score in boosted])
    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score == 0:
        normalized = np.ones_like(scores)
    else:
        normalized = (scores - min_score) / (max_score - min_score)

    reranked = sorted(
        zip([doc for doc, _ in boosted], normalized),
        key=lambda x: x[1],
        reverse=True
    )[:rerankK]

    return [(doc, float(score)) for doc, score in reranked]

@app.post("/search")
def search(request: SearchRequest):
    start_time = time.time()

    candidates = retrieve(request.query, request.k)

    if request.rerank:
        final_results = rerank_results(request.query, candidates, request.rerankK)
        reranked_flag = True
    else:
        final_results = candidates[:request.rerankK]
        reranked_flag = False

    results = [
        {
            "id": doc["id"],
            "score": round(score, 4),
            "content": doc["content"],
            "metadata": doc["metadata"]
        }
        for doc, score in final_results
    ]

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": results,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
