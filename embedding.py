import re
import json
import requests
import numpy as np
from collections import defaultdict, Counter

EMBEDDING_URL = "http://127.0.0.1:8081/v1/embeddings"
BATCH_SIZE = 16

session = requests.Session()
session.trust_env = False

def load_tables(path):
    tables = []
    current = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("TABLE:"):
                if current:
                    current["text"] = " ; ".join(
                        dict.fromkeys(current["parts"])
                    )
                    del current["parts"]
                    tables.append(current)

                current = {
                    "table": line.split(":", 1)[1].strip(),
                    "code": "",
                    "sujet": "",
                    "parts": []
                }

            elif current:
                if line.startswith("CODE:"):
                    current["code"] = line.split(":", 1)[1].strip()

                elif line.startswith("SECTION:"):
                    current["parts"].append(
                        line.split(":", 1)[1].strip()
                    )

                elif line.startswith("SUJET:"):
                    sujet = line.split(":", 1)[1].strip()
                    current["sujet"] = sujet
                    current["parts"].append(sujet)

                elif line.startswith(("CONTEXTE:", "POPULATION_CONTEXT:")):
                    current["parts"].append(
                        line.split(":", 1)[1].strip()
                    )
                    
                elif line.startswith("libellé:"):
                    current["parts"].append(
                        line.split(":", 1)[1].strip()
                    )

    if current:
        current["text"] = " ; ".join(
            dict.fromkeys(current["parts"])
        )
        del current["parts"]
        tables.append(current)

    return tables

def embed(texts):
    response = session.post(
        EMBEDDING_URL,
        json={
            "input": texts,
            "model": "local",
            "encoding_format": "float"
        },
        timeout=(5, 120)
    )

    response.raise_for_status()

    data = sorted(
        response.json()["data"],
        key=lambda x: x.get("index", 0)
    )

    return np.array(
        [x["embedding"] for x in data],
        dtype=np.float32
    )


def family_name(table):
    patterns = [
        r"^(cti_viii_)d\d+(_.*)$",
        r"^(cti_vii_)d\d+(_.*)?$",
        r"^(cti_ii_1_)d\d+(_.*)$"
    ]

    for pattern in patterns:
        m = re.fullmatch(pattern, table)

        if m:
            return f"{m.group(1)}d*{m.group(2) or ''}"

    return table

raw_entries = load_tables("semantic_schema.txt")


# -------------------------------------------------
# Regroupement D1...D11
# -------------------------------------------------

groups = defaultdict(list)

for entry in raw_entries:
    groups[family_name(entry["table"])].append(entry)


entries = []

for family, members in groups.items():

    # Sujet le plus fréquent dans la famille
    sujet = Counter(
        x["sujet"] for x in members
    ).most_common(1)[0][0]

    # VIII-D6-7 -> VIII-D*-7
    code = re.sub(
        r"D\d+",
        "D*",
        members[0]["code"],
        count=1
    )

    entries.append({
    "table": family,
    "code": code,
    "sujet": sujet,
    "members": [x["table"] for x in members],
    "text": max(
        (x["text"] for x in members),
        key=len
        )
    })


print(f"{len(raw_entries)} tables physiques")
print(f"{len(entries)} entrées sémantiques après regroupement")


# -------------------------------------------------
# Calcul des embeddings
# -------------------------------------------------

texts = [x["text"] for x in entries]

all_vectors = []

for start in range(0, len(texts), BATCH_SIZE):

    batch = texts[start:start + BATCH_SIZE]

    print(
        f"Embedding {start + 1}-"
        f"{start + len(batch)} / {len(texts)}",
        flush=True
    )

    all_vectors.append(embed(batch))


vectors = np.vstack(all_vectors)

# Normalisation pour similarité cosinus
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
norms[norms == 0] = 1
vectors /= norms


# -------------------------------------------------
# Sauvegarde
# -------------------------------------------------

np.save("table_embeddings.npy", vectors)

with open("table_entries.json", "w", encoding="utf-8") as f:
    json.dump(
        entries,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nTerminé.")
print(f"{len(entries)} entrées indexées")
print(f"Dimension des embeddings : {vectors.shape[1]}")