import requests
import numpy as np
import json
import re
import sqlite3

EMBEDDING_URL = "http://127.0.0.1:8081/v1/embeddings"

session = requests.Session()
session.trust_env = False

vectors = np.load("table_embeddings.npy")

with open("table_entries.json", "r", encoding="utf-8") as f:
    entries = json.load(f)


def embed_question(question):
    response = session.post(
        EMBEDDING_URL,
        json={
            "input": question
        }
    )
    response.raise_for_status()

    vector = np.array(
        response.json()["data"][0]["embedding"],
        dtype=np.float32
    )

    vector /= np.linalg.norm(vector)

    return vector


def search_tables(question, top_k=8):
    q = embed_question(question)

    # Les vecteurs du schéma sont déjà normalisés
    scores = vectors @ q

    indexes = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in indexes:
        results.append({
            "table": entries[i]["table"],
            "code": entries[i]["code"],
            "sujet": entries[i]["sujet"],
            "text": entries[i]["text"],
            "members": entries[i]["members"],
            "score": float(scores[i])
        })

    return results

