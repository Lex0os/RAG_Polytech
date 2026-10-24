import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from tables_search import search_tables


# ---------------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------------

POLYTECH_SCHOOLS = (
    "ANGERS",
    "ANNECY_CHAMBERY",
    "CLERMONT",
    "DIJON",
    "GRENOBLE",
    "LILLE",
    "LYON",
    "MARSEILLE",
    "MONTPELLIER",
    "NANCY",
    "NANTES",
    "NICE",
    "ORLEANS",
    "SACLAY",
    "SORBONNE",
    "TOURS",
)

MAX_SELECTED_FAMILIES = 3
MAX_FINAL_ROWS = 100
MAX_DIMENSION_VALUES_IN_PROMPT = 20
MAX_ROUTER_COLUMNS = 5

# Rappel lexical BM25.
# k1 contrôle la saturation de la fréquence des termes.
# b contrôle la normalisation par longueur des documents.
BM25_RECALL_K = 8
BM25_K1 = 1.5
BM25_B = 0.75

# Alias conservé pour compatibilité avec un éventuel code externe.
LEXICAL_RECALL_K = BM25_RECALL_K

MAX_HYBRID_CANDIDATES = 40


# ---------------------------------------------------------------------------
# Utilitaires de texte / schéma
# ---------------------------------------------------------------------------

def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(
        char for char in text
        if not unicodedata.combining(char)
    )


def text_token_list(text):
    """
    Tokenisation lexicale conservant les répétitions.

    BM25 a besoin de la fréquence réelle d'un terme dans un document,
    contrairement aux anciens calculs basés uniquement sur présence/absence.
    """
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if len(token) >= 3
    ]


def text_tokens(text):
    """
    Version en ensemble utilisée par les autres garde-fous lexicaux du moteur.
    """
    return set(text_token_list(text))


def load_schema_blocks(path):
    """
    Charge semantic_schema.txt dans un dictionnaire :
        nom_table -> bloc sémantique complet
    """
    content = Path(path).read_text(encoding="utf-8")
    schemas = {}

    for block in re.split(r"(?m)(?=^TABLE:\s+)", content):
        block = block.strip()

        if not block.startswith("TABLE:"):
            continue

        first_line = block.splitlines()[0]
        table = first_line.split(":", 1)[1].strip()
        schemas[table] = block

    return schemas


def schema_header_value(schema, key):
    match = re.search(
        rf"(?m)^{re.escape(key)}:\s*(.+)$",
        schema,
    )
    return match.group(1).strip() if match else ""


def schema_columns(schema):
    """
    Retourne uniquement les colonnes de COLONNES METIER.
    """
    marker = "COLONNES METIER:"
    if marker not in schema:
        return []

    section = schema.split(marker, 1)[1]
    columns = []

    for line in section.splitlines():
        match = re.match(
            r"^\s*-\s+([a-zA-Z0-9_]+)\s+(TEXT|REAL|INTEGER)\b",
            line,
        )
        if match:
            columns.append(match.group(1))

    return columns


def parse_business_columns(schema):
    """
    Extrait les métadonnées utiles des colonnes métier sans transporter
    tout le texte du schéma dans les prompts.
    """
    marker = "COLONNES METIER:"
    if marker not in schema:
        return []

    lines = schema.split(marker, 1)[1].splitlines()
    result = []
    current = None

    for line in lines:
        column_match = re.match(
            r"^\s*-\s+([a-zA-Z0-9_]+)\s+(TEXT|REAL|INTEGER)\b",
            line,
        )

        if column_match:
            if current:
                result.append(current)

            current = {
                "name": column_match.group(1),
                "sql_type": column_match.group(2),
                "label": "",
                "type": "",
                "unit": "",
                "aggregation": "",
                "description": "",
            }
            continue

        if current is None:
            continue

        stripped = line.strip()

        mappings = {
            "libellé:": "label",
            "type:": "type",
            "unité:": "unit",
            "agrégation:": "aggregation",
            "description:": "description",
        }

        for prefix, field in mappings.items():
            if stripped.startswith(prefix):
                current[field] = stripped.split(":", 1)[1].strip()
                break

    if current:
        result.append(current)

    return result


def get_dimension_values(schema, dimension):
    """
    Lit les valeurs observées de row_name / row_group dans le schéma complet.
    """
    lines = schema.splitlines()

    for index, line in enumerate(lines):
        if not re.match(
            rf"^\s*-\s+{re.escape(dimension)}\s+",
            line,
        ):
            continue

        for next_line in lines[index + 1:index + 4]:
            stripped = next_line.strip()

            if stripped.startswith("valeurs observées"):
                if ":" not in stripped:
                    return []

                values_text = stripped.split(":", 1)[1].strip()
                return [
                    value.strip().strip('"')
                    for value in values_text.split("|")
                    if value.strip()
                ]

            if re.match(r"^-\s+", stripped):
                break

        return []

    return []


def candidate_context(candidate, schemas):
    """
    Le CONTEXTE vient du schéma détaillé et non d'une règle codée en dur.
    """
    contexts = []

    for table in candidate.get("members", []):
        schema = schemas.get(table)

        if not schema:
            continue

        context = schema_header_value(schema, "CONTEXTE")

        if context and context not in contexts:
            contexts.append(context)

        if len(contexts) >= 2:
            break

    return " | ".join(contexts)


def candidate_column_hints(question, candidate, schemas, limit=MAX_ROUTER_COLUMNS):
    """
    Fournit quelques libellés de colonnes au routeur.
    C'est une aide lexicale générique, pas une règle métier.
    """
    q_tokens = text_tokens(question)
    scored = []

    for table in candidate.get("members", [])[:2]:
        schema = schemas.get(table)

        if not schema:
            continue

        for column in parse_business_columns(schema):
            searchable = " ".join(
                [
                    column["name"],
                    column["label"],
                    column["type"],
                    column["unit"],
                ]
            )

            overlap = len(q_tokens & text_tokens(searchable))
            scored.append(
                (
                    overlap,
                    column["label"] or column["name"],
                )
            )

    # Les colonnes lexicalement proches passent d'abord.
    scored.sort(key=lambda item: item[0], reverse=True)

    result = []
    for _, label in scored:
        if label and label not in result:
            result.append(label)

        if len(result) >= limit:
            break

    return result


def candidate_type_hints(candidate, schemas):
    """
    Résumé très compact des types/agrégations disponibles.
    Aide GPT à savoir si une famille contient directement les ingrédients
    d'un calcul dérivé, sans ajouter de règle métier.
    """
    types = []
    aggregations = []

    for table in candidate.get("members", [])[:2]:
        schema = schemas.get(table)

        if not schema:
            continue

        for column in parse_business_columns(schema):
            if column["type"] and column["type"] not in types:
                types.append(column["type"])

            if (
                column["aggregation"]
                and column["aggregation"] not in aggregations
            ):
                aggregations.append(column["aggregation"])

    parts = []

    if types:
        parts.append("types=" + ", ".join(types[:5]))

    if aggregations:
        parts.append(
            "agrégations=" + ", ".join(aggregations[:4])
        )

    return " | ".join(parts)


def compact_dimension_line(schema, dimension):
    values = get_dimension_values(schema, dimension)

    if not values:
        return f"{dimension}: présent, aucune valeur observée fournie"

    shown = values[:MAX_DIMENSION_VALUES_IN_PROMPT]
    suffix = ""

    if len(values) > len(shown):
        suffix = f" | ... ({len(values)} valeurs au total)"

    return f"{dimension}: " + " | ".join(shown) + suffix


def compact_table_schema(schema):
    """
    Version compacte d'un bloc de semantic_schema.txt pour GPT.
    Les informations sémantiques utiles sont conservées :
    sujet, contexte, dimensions, libellé, type, unité, agrégation.
    """
    table = schema_header_value(schema, "TABLE")
    code = schema_header_value(schema, "CODE")
    section = schema_header_value(schema, "SECTION")
    subject = schema_header_value(schema, "SUJET")
    context = schema_header_value(schema, "CONTEXTE")

    parts = [
        f"TABLE: {table}",
    ]

    if code:
        parts.append(f"CODE: {code}")

    if section:
        parts.append(f"SECTION: {section}")

    if subject:
        parts.append(f"SUJET: {subject}")

    if context:
        parts.append(f"CONTEXTE: {context}")

    parts.append("DIMENSIONS:")
    parts.append("- school_key: identifiant de l'école")
    parts.append("- year: année / campagne")
    parts.append("- " + compact_dimension_line(schema, "row_name"))
    parts.append("- " + compact_dimension_line(schema, "row_group"))

    parts.append("COLONNES METIER:")

    for column in parse_business_columns(schema):
        metadata = []

        if column["label"]:
            metadata.append(f"libellé={column['label']}")

        if column["type"]:
            metadata.append(f"type={column['type']}")

        if column["unit"]:
            metadata.append(f"unité={column['unit']}")

        if column["aggregation"]:
            metadata.append(f"agrégation={column['aggregation']}")

        # Une description spécifique est utile ; les descriptions génériques
        # répétitives n'apportent rien au modèle.
        description = column["description"]
        generic_descriptions = (
            "Valeur extraite du formulaire",
            "Effectif ou quantité déclarée",
            "Nombre de répondants associé",
            "Pourcentage déclaré dans le formulaire",
            "Salaire médian annuel extrait",
        )

        if (
            description
            and not any(
                description.startswith(prefix)
                for prefix in generic_descriptions
            )
        ):
            metadata.append(f"description={description}")

        suffix = " | ".join(metadata)
        parts.append(
            f"- {column['name']} {column['sql_type']}"
            + (f" | {suffix}" if suffix else "")
        )

    return "\n".join(parts)


def compact_family_schema(entry, schemas):
    """
    Pour une famille D*, un seul membre reçoit le détail complet.
    Les autres membres reçoivent seulement leur liste de colonnes afin
    d'éviter de répéter 10 fois le même schéma.
    """
    members = [
        table
        for table in entry.get("members", [])
        if table in schemas
    ]

    if not members:
        return ""

    family_name = entry["table"]
    subject = entry.get("sujet", "")
    context = candidate_context(entry, schemas)

    header = [
        f"FAMILLE: {family_name}",
        f"SUJET: {subject}",
    ]

    if context:
        header.append(f"CONTEXTE: {context}")

    # Une table normale : bloc compact complet.
    if "*" not in family_name or len(members) == 1:
        blocks = [
            compact_table_schema(schemas[table])
            for table in members
        ]
        return "\n".join(header) + "\n\n" + "\n\n".join(blocks)

    representative = members[0]

    parts = [
        "\n".join(header),
        "",
        "SCHÉMA SÉMANTIQUE REPRÉSENTATIF:",
        compact_table_schema(schemas[representative]),
        "",
        "TABLES PHYSIQUES DE LA FAMILLE ET COLONNES DISPONIBLES:",
    ]

    for table in members:
        parts.append(
            f"- {table}: {', '.join(schema_columns(schemas[table]))}"
        )

    # Les dimensions peuvent varier légèrement entre membres.
    row_names = []
    row_groups = []

    for table in members:
        for value in get_dimension_values(schemas[table], "row_name"):
            if value not in row_names:
                row_names.append(value)

        for value in get_dimension_values(schemas[table], "row_group"):
            if value not in row_groups:
                row_groups.append(value)

    if row_names:
        shown = row_names[:MAX_DIMENSION_VALUES_IN_PROMPT]
        extra = (
            f" | ... ({len(row_names)} valeurs au total)"
            if len(row_names) > len(shown)
            else ""
        )
        parts.append(
            "ROW_NAME observés dans la famille: "
            + " | ".join(shown)
            + extra
        )

    if row_groups:
        shown = row_groups[:MAX_DIMENSION_VALUES_IN_PROMPT]
        extra = (
            f" | ... ({len(row_groups)} valeurs au total)"
            if len(row_groups) > len(shown)
            else ""
        )
        parts.append(
            "ROW_GROUP observés dans la famille: "
            + " | ".join(shown)
            + extra
        )

    return "\n".join(parts)



def _virtual_family_name(table):
    """
    Regroupe structurellement les tables D1/D2/... sous une famille D*.
    Exemple :
        cti_viii_d3_7 -> cti_viii_d*_7

    Ce regroupement ne porte aucune sémantique métier.
    """
    return re.sub(
        r"_d\d+(?=_|$)",
        "_d*",
        table,
        flags=re.IGNORECASE,
    )


def _family_members_from_schemas(family, schemas):
    """
    Résout les membres physiques d'une famille à partir du semantic_schema.
    """
    if "*" not in family:
        return [family] if family in schemas else []

    pattern = re.compile(
        "^"
        + re.escape(family).replace(r"\*", r"\d+")
        + "$",
        flags=re.IGNORECASE,
    )

    return sorted(
        [
            table
            for table in schemas
            if pattern.match(table)
        ],
        key=lambda value: (
            int(re.search(r"_d(\d+)(?:_|$)", value, re.I).group(1))
            if re.search(r"_d(\d+)(?:_|$)", value, re.I)
            else 10**9,
            value,
        ),
    )


def _candidate_searchable_text(candidate, schemas):
    parts = [
        candidate.get("table", ""),
        candidate.get("code", ""),
        candidate.get("sujet", ""),
    ]

    members = candidate.get("members", [])

    for table in members:
        schema = schemas.get(table, "")

        if not schema:
            continue

        parts.extend(
            [
                schema_header_value(schema, "SUJET"),
                schema_header_value(schema, "CONTEXTE"),
                schema_header_value(schema, "SECTION"),
                " ".join(get_dimension_values(schema, "row_name")),
                " ".join(get_dimension_values(schema, "row_group")),
            ]
        )

        for column in parse_business_columns(schema):
            parts.extend(
                [
                    column.get("name", ""),
                    column.get("label", ""),
                    column.get("type", ""),
                    column.get("unit", ""),
                    column.get("description", ""),
                ]
            )

    return " ".join(
        str(part)
        for part in parts
        if part
    )


def _all_schema_family_candidates(schemas):
    """
    Construit un index lexical logique directement depuis semantic_schema.
    Les groupes D* sont créés automatiquement à partir des noms de tables.
    """
    grouped = {}

    for table, schema in schemas.items():
        family = _virtual_family_name(table)

        grouped.setdefault(
            family,
            {
                "table": family,
                "members": [],
                "subjects": [],
                "codes": [],
            },
        )

        grouped[family]["members"].append(table)

        subject = schema_header_value(schema, "SUJET")
        code = schema_header_value(schema, "CODE")

        if subject and subject not in grouped[family]["subjects"]:
            grouped[family]["subjects"].append(subject)

        if code and code not in grouped[family]["codes"]:
            grouped[family]["codes"].append(code)

    candidates = []

    for family, data in grouped.items():
        members = sorted(
            data["members"],
            key=lambda value: (
                int(re.search(r"_d(\d+)(?:_|$)", value, re.I).group(1))
                if re.search(r"_d(\d+)(?:_|$)", value, re.I)
                else 10**9,
                value,
            ),
        )

        # Un seul membre => conserver le vrai nom physique.
        logical_name = family if len(members) > 1 else members[0]

        # Le sujet le plus fréquent est généralement le sujet de la famille.
        subject_counts = {}
        for subject in data["subjects"]:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

        subject = (
            max(
                subject_counts,
                key=lambda value: (
                    subject_counts[value],
                    len(value),
                ),
            )
            if subject_counts
            else ""
        )

        candidates.append(
            {
                "table": logical_name,
                "code": data["codes"][0] if data["codes"] else "",
                "sujet": subject,
                "members": members,
                "retrieval_source": "bm25_schema",
            }
        )

    return candidates


def build_bm25_schema_index(schemas):
    """
    Construit une fois l'index BM25 du semantic_schema.

    Chaque famille logique constitue un document. Le texte indexé est celui de
    _candidate_searchable_text(), donc il inclut notamment :
      - table / code / sujet ;
      - section / contexte ;
      - row_name / row_group observés ;
      - noms, libellés, types, unités et descriptions des colonnes.

    L'index est volontairement indépendant des embeddings : il sert uniquement
    de rappel lexical complémentaire.
    """
    candidates = _all_schema_family_candidates(schemas)

    documents = []
    document_frequency = Counter()
    total_length = 0

    for candidate in candidates:
        searchable = _candidate_searchable_text(
            candidate,
            schemas,
        )

        tokens = text_token_list(searchable)
        term_frequencies = Counter(tokens)
        length = len(tokens)

        documents.append(
            {
                "candidate": candidate,
                "term_frequencies": term_frequencies,
                "length": length,
                # Normalisation lexicale légère des pluriels pour retrouver
                # les catégories au singulier comme au pluriel. Aucune liste
                # de libellés métier et aucune décision sur la source.
                "dimension_tokens": {
                    token[:-1] if len(token) > 4 and token.endswith("s") else token
                    for token in text_tokens(" ".join(
                        value
                        for table in candidate.get("members", [])
                        for dimension in ("row_name", "row_group")
                        for value in get_dimension_values(schemas.get(table, ""), dimension)
                    ))
                },
            }
        )

        total_length += length

        for token in term_frequencies:
            document_frequency[token] += 1

    document_count = len(documents)
    average_document_length = (
        total_length / document_count
        if document_count
        else 0.0
    )

    return {
        "documents": documents,
        "document_frequency": document_frequency,
        "document_count": document_count,
        "average_document_length": average_document_length,
    }


def _bm25_idf(term, bm25_index):
    """
    IDF de Robertson/Sparck Jones avec variante positive courante :

        log(1 + (N - df + 0.5) / (df + 0.5))

    Le +1 dans le log évite un IDF négatif pour les termes très fréquents.
    """
    document_count = bm25_index["document_count"]
    document_frequency = bm25_index["document_frequency"].get(
        term,
        0,
    )

    if document_count <= 0 or document_frequency <= 0:
        return 0.0

    return math.log(
        1.0
        + (
            document_count
            - document_frequency
            + 0.5
        )
        / (
            document_frequency
            + 0.5
        )
    )


def bm25_schema_candidates(
    question,
    bm25_index,
    top_k=BM25_RECALL_K,
    k1=BM25_K1,
    b=BM25_B,
):
    """
    Rappel lexical BM25 sur le semantic_schema.

    Formule Okapi BM25 utilisée pour chaque terme q de la question :

        IDF(q) * (
            tf(q, D) * (k1 + 1)
            /
            (
                tf(q, D)
                + k1 * (
                    1 - b
                    + b * |D| / avgdl
                )
            )
        )

    Contrairement à l'ancien score maison :
      - la fréquence du terme dans le document est prise en compte ;
      - les documents longs sont normalisés ;
      - SequenceMatcher n'intervient plus dans le retrieval lexical.

    Les embeddings restent utilisés en parallèle ; BM25 sert seulement à
    ajouter des candidats lexicaux que les embeddings pourraient manquer.
    """
    query_tokens = text_token_list(question)

    if not query_tokens:
        return []

    documents = bm25_index.get("documents", [])
    average_document_length = bm25_index.get(
        "average_document_length",
        0.0,
    )

    if not documents or average_document_length <= 0:
        return []

    # La fréquence dans la question est conservée : un terme répété
    # explicitement par l'utilisateur peut donc peser davantage.
    query_frequencies = Counter(query_tokens)
    scored = []

    for document in documents:
        term_frequencies = document["term_frequencies"]
        document_length = document["length"]

        score = 0.0
        matched_terms = []

        for term, query_frequency in query_frequencies.items():
            term_frequency = term_frequencies.get(
                term,
                0,
            )

            if term_frequency <= 0:
                continue

            matched_terms.append(term)

            idf = _bm25_idf(
                term,
                bm25_index,
            )

            length_normalization = (
                1.0
                - b
                + b
                * document_length
                / average_document_length
            )

            denominator = (
                term_frequency
                + k1 * length_normalization
            )

            term_score = (
                idf
                * (
                    term_frequency
                    * (k1 + 1.0)
                    / denominator
                )
            )

            score += (
                query_frequency
                * term_score
            )

        if score <= 0:
            continue

        item = dict(
            document["candidate"]
        )

        # bm25_score est le nom canonique.
        item["bm25_score"] = score

        # Alias temporaire pour ne pas casser une UI déjà adaptée au champ
        # lexical_score. Les deux valeurs sont identiques dans cette version.
        item["lexical_score"] = score

        item["lexical_terms"] = sorted(
            set(matched_terms)
        )
        item["retrieval_source"] = "bm25_schema"

        scored.append(item)

    scored.sort(
        key=lambda item: item["bm25_score"],
        reverse=True,
    )

    # Rappel complémentaire des tables portant plusieurs catégories demandées.
    # Les libellés viennent du schéma, sans liste de catégories métier.
    # Cela rend visibles les effectifs stockés en lignes plutôt qu'en colonnes.
    dimension_matches = []
    category_query_tokens = {
        token[:-1] if len(token) > 4 and token.endswith("s") else token
        for token in query_tokens
    }
    for document in documents:
        overlap = category_query_tokens & document.get("dimension_tokens", set())
        if len(overlap) >= 2:
            item = dict(document["candidate"])
            item["lexical_terms"] = sorted(overlap)
            item["retrieval_source"] = "schema_dimensions"
            dimension_matches.append((len(overlap), item))
    dimension_matches.sort(key=lambda pair: (-pair[0], pair[1]["table"]))
    result = [item for _, item in dimension_matches[:top_k]]
    names = {item["table"] for item in result}
    result.extend(item for item in scored[:top_k] if item["table"] not in names)
    return result


def lexical_schema_candidates(
    question,
    schemas,
    top_k=LEXICAL_RECALL_K,
):
    """
    Compatibilité avec l'ancienne API.

    Cette fonction reconstruit un index à chaque appel et ne doit donc pas être
    utilisée dans DataQueryEngine.ask(). Le moteur principal utilise
    bm25_schema_candidates() avec un index pré-calculé dans __init__.
    """
    bm25_index = build_bm25_schema_index(
        schemas
    )

    return bm25_schema_candidates(
        question,
        bm25_index,
        top_k=top_k,
    )


def merge_hybrid_candidates(
    embedding_candidates,
    lexical_candidates,
    schemas,
):
    """
    Fusionne les candidats embedding et rappel BM25 sans doublon.
    Les embeddings restent la base ; le rappel BM25 ajoute les familles
    qui auraient pu être manquées sémantiquement.
    """
    merged = []
    by_name = {}

    def add(candidate, source):
        item = dict(candidate)
        name = item.get("table")

        if not name:
            return

        members = [
            table
            for table in item.get("members", [])
            if table in schemas
        ]

        if not members:
            members = _family_members_from_schemas(
                name,
                schemas,
            )

        if not members:
            return

        item["members"] = members

        if name in by_name:
            existing = by_name[name]

            if item.get("bm25_score") is not None:
                existing["bm25_score"] = item["bm25_score"]

                # Alias conservé pour compatibilité avec l'interface existante.
                existing["lexical_score"] = item["bm25_score"]

                existing["lexical_terms"] = item.get(
                    "lexical_terms",
                    [],
                )
            elif item.get("lexical_score") is not None:
                existing["lexical_score"] = item["lexical_score"]
                existing["lexical_terms"] = item.get(
                    "lexical_terms",
                    [],
                )

            sources = set(
                str(existing.get("retrieval_source", ""))
                .split("+")
            )
            sources.add(source)
            sources.discard("")
            existing["retrieval_source"] = "+".join(
                sorted(sources)
            )
            return

        item["retrieval_source"] = source
        by_name[name] = item
        merged.append(item)

    for candidate in embedding_candidates:
        add(candidate, "embedding")

    for candidate in lexical_candidates:
        add(candidate, "bm25_schema")

    return merged[:MAX_HYBRID_CANDIDATES]


# ---------------------------------------------------------------------------
# Routage par GPT
# ---------------------------------------------------------------------------

def normalize_candidates(candidates, schemas):
    """
    Nettoyage purement structurel des résultats d'embedding.
    """
    result = []

    for candidate in candidates:
        item = dict(candidate)

        members = [
            table
            for table in item.get("members", [])
            if table in schemas
        ]

        if not members and item.get("table") in schemas:
            members = [item["table"]]

        if not members and item.get("table"):
            members = _family_members_from_schemas(
                item["table"],
                schemas,
            )

        if not members:
            continue

        item["members"] = members
        result.append(item)

    return result


def build_candidate_coverage(connection, candidates, schemas):
    """Disponibilité par campagne, sans choisir de mesure ni de source métier."""
    table_stats = {}
    result = {}
    placeholders = ",".join("?" for _ in POLYTECH_SCHOOLS)
    for candidate in candidates:
        campaigns = {}
        for table in candidate.get("members", []):
            if table not in table_stats:
                columns = [column["name"] for column in parse_business_columns(schemas.get(table, ""))
                           if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column["name"])]
                if not columns or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
                    continue
                populated = " OR ".join(
                    f'("{column}" IS NOT NULL AND TRIM(CAST("{column}" AS TEXT)) <> \'\')'
                    for column in columns
                )
                table_stats[table] = connection.execute(
                    f'SELECT DISTINCT year, school_key FROM "{table}" '
                    f'WHERE school_key IN ({placeholders}) AND ({populated})',
                    POLYTECH_SCHOOLS,
                ).fetchall()
            for year, school in table_stats[table]:
                campaigns.setdefault(year, set()).add(school)
        result[candidate["table"]] = "; ".join(
            f"campagne {year} : {len(schools)}/16 écoles"
            for year, schools in sorted(campaigns.items(), key=lambda item: str(item[0]), reverse=True)
        ) or "aucune valeur métier renseignée dans le réseau"
    return result


def route_tables(llm, question, candidates, schemas, coverage=None):
    """
    Un seul reranking GPT.
    Les scores embedding ne sont volontairement pas envoyés au modèle.
    """
    candidate_blocks = []

    for candidate in candidates:
        context = candidate_context(candidate, schemas)
        hints = candidate_column_hints(
            question,
            candidate,
            schemas,
        )
        type_hints = candidate_type_hints(
            candidate,
            schemas,
        )

        block = [
            f"TABLE: {candidate['table']}",
            f"SUJET: {candidate.get('sujet', '')}",
        ]

        if coverage is not None:
            block.append("DISPONIBILITÉ RÉSEAU : " + coverage.get(candidate["table"], "inconnue"))
        for dimension in ("row_name", "row_group"):
            values = list(dict.fromkeys(
                value for table in candidate.get("members", [])
                for value in get_dimension_values(schemas.get(table, ""), dimension)
            ))
            if values:
                block.append(f"CATÉGORIES {dimension}: " + " | ".join(values[:MAX_DIMENSION_VALUES_IN_PROMPT]))

        if context:
            block.append(f"CONTEXTE: {context}")

        if hints:
            block.append(
                "COLONNES UTILES POSSIBLES: "
                + " | ".join(hints)
            )

        if type_hints:
            block.append(
                "MÉTADONNÉES: " + type_hints
            )

        lexical_terms = candidate.get(
            "lexical_terms",
            [],
        )
        if lexical_terms:
            block.append(
                "TERMES DISTINCTIFS DE LA QUESTION PRÉSENTS: "
                + " | ".join(lexical_terms[:8])
            )

        candidate_blocks.append("\n".join(block))

    allowed = [candidate["table"] for candidate in candidates]

    response_schema = {
        "type": "object",
        "properties": {
            "families": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": allowed,
                },
                "minItems": 1,
                "maxItems": MAX_SELECTED_FAMILIES,
            }
        },
        "required": ["families"],
        "additionalProperties": False,
    }

    prompt = f"""
QUESTION:
{question}

FAMILLES CANDIDATES:
{chr(10).join(chr(10) + block for block in candidate_blocks)}

Choisis uniquement la ou les familles nécessaires pour répondre.

ORDRE DE DÉCISION OBLIGATOIRE:
1. Définis la population, la période et le niveau de synthèse demandés.
2. Vérifie les ingrédients du calcul dans chaque candidat : une catégorie
   stockée dans row_name/row_group est aussi utilisable qu'une colonne dédiée.
   Pour une synthèse globale, des effectifs sommables des deux catégories
   permettent le calcul ; des pourcentages sans dénominateurs ne le permettent pas.
3. Écarte les sources insuffisantes si une source de la population demandée
   fournit ces ingrédients. Ne sélectionne pas une rubrique de pourcentages
   seuls pour conclure « non calculable » lorsque les effectifs compatibles
   figurent dans une autre famille candidate. Un quotient de sommes d'effectifs
   est une reconstruction normale, pas un motif pour préférer les pourcentages.
4. Entre sources calculables et sémantiquement compatibles, compare la couverture.
   Distingue une contrainte explicitement demandée d'un détail ajouté par toi :
   pour une question générale sur le réseau sans horizon précis, ne restreins
   pas arbitrairement le calcul à un horizon propre à une source moins couvrante.
   Parmi les observations récentes du même concept et de la même population,
   privilégie la couverture, puis conserve et annonce la période réelle de la
   source choisie. Cela n'autorise ni à mélanger les horizons ni à assimiler
   deux définitions différentes. Un horizon explicitement demandé reste prioritaire.
5. La proximité du titre et le nombre minimal de familles départagent seulement
   les solutions permettant effectivement de répondre. Avant de rendre families,
   vérifie que ton choix permet le calcul global et qu'aucun candidat compatible
   plus complet n'a été écarté au seul motif que son titre paraît moins direct.

Principes:
- sans école explicitement demandée ni ventilation demandée (« par école »,
  comparaison ou classement), vise un chiffre global du réseau, pas une liste
  de chiffres par école. Pour un ratio ou une proportion globale, recherche les
  effectifs nécessaires même si une autre rubrique nomme directement la répartition.
  Des pourcentages seuls ne suffisent pas à agréger plusieurs populations ;
- parmi les sources permettant le même calcul sur la même population et période,
  privilégie celle qui couvre le plus d'écoles du réseau à la campagne pertinente.
  La DISPONIBILITÉ indique des écoles avec au moins une valeur métier, pas la
  validité du couple numérateur/dénominateur : le SQL doit vérifier celle-ci.
  Ne choisis pas une population voisine ou une campagne ancienne pour maximiser
  artificiellement la couverture. N'assemble pas CTI et DEFI pour compter deux
  fois les mêmes personnes. La préférence pour un titre directement similaire
  ne prime pas sur les ingrédients nécessaires au calcul global ;
- une demande visant une seule école n'impose aucune préférence pour un type
  de document : choisis la source exacte et exploitable pour cette école.
  Pour le réseau, ne te limite pas aux écoles d'une source moins couvrante si
  une alternative récente permet le calcul sur la population demandée. N'impose
  aucune priorité CTI/DEFI indépendante des données. Une disponibilité de 16
  écoles est un indice à vérifier sur les couples utiles, pas une garantie que
  toutes les observations nécessaires sont renseignées ;
- base-toi sur le sens exact du SUJET, du CONTEXTE, des colonnes et
  de leurs métadonnées ;
- accorde une importance forte aux termes distinctifs explicitement présents
  à la fois dans la question et dans le schéma d'une famille. Si un terme rare
  de la question correspond directement au SUJET/à une colonne d'une famille,
  ne le remplace pas par une famille voisine portant sur une autre population.
  Cette proximité lexicale ne suffit pas si les ingrédients du calcul manquent ;
- préfère une famille dont le SUJET représente directement le concept
  demandé à une table plus large où il faudrait reconstruire ce concept
  à partir de nombreuses catégories, uniquement si les deux permettent le calcul
  demandé. Des effectifs par catégorie priment sur une répartition en pourcentages
  sans poids pour un ratio global de personnes ;
- pour une mesure dérivée (taux, ratio, moyenne...), une famille décrivant
  directement la population demandée et contenant les ingrédients du calcul
  est préférable à une table générale seulement voisine du sujet ;
- une restriction absente de la question ne doit pas être ajoutée ;
- une question générale doit privilégier la population générale ;
- pour une synthèse générale portant sur des apprenants ou diplômés, ne limite
  pas implicitement la population à une voie : inclure FISE et FISA + FISEA
  lorsqu'elles sont présentes et compatibles avec la question, sur tous les
  membres concernés. Une restriction à une seule voie doit être explicitement
  demandée. Ces voies peuvent être des colonnes d'une même famille ; leur
  combinaison n'exige donc pas nécessairement plusieurs familles ;
- respecte les périodes et populations mentionnées dans la question ;
- distingue la population (apprenants inscrits, diplômés, répondants,
  diplômés en emploi, personnels de l'école), la promotion et l'horizon de
  mesure. Un délai pour trouver un emploi n'est pas une situation observée
  à six mois. Une création d'entreprise n'établit pas un statut juridique
  d'autoentrepreneur. Un cadre parmi les personnels n'est pas un diplômé cadre ;
- pour des effectifs d'apprentis actuellement inscrits, privilégie les effectifs
  d'inscrits par voie, pas les seuls diplômés ni les contrats signés dans l'année.
  Pour une durée de recherche du premier emploi, privilégie la table de délais
  plutôt qu'un nombre de diplômés encore en recherche à une date donnée ;
- respecte les bornes exactes des tranches : « 6 mois ou plus » inclut six mois
  et ne mesure pas exactement « plus de 6 mois ». Si le concept exact manque,
  choisis les données les plus pertinentes sans les présenter comme équivalentes ;
- convention labels de développement durable : une demande générale de labels
  ou de « label DDRS » couvre la rubrique cti_v_5, pas seulement le label DD&RS ;
- choisis l'ensemble MINIMAL de familles suffisant pour répondre ;
- pour une proportion issue d'une enquête, recherche d'abord dans la famille
  de l'indicateur le nombre de réponses associé (type response_count). Si la
  question ne demande pas explicitement un autre dénominateur, sélectionne
  cette famille seule lorsqu'elle fournit mesure et réponses compatibles.
  N'ajoute pas une famille de population générale ou de statut voisin pour
  remplacer ce dénominateur local. Un autre dénominateur exige une définition
  explicitement demandée et des populations/périodes réellement compatibles ;
- n'ajoute pas de table descriptive uniquement pour répéter une information déjà
  connue via school_key, year ou une autre dimension de la table principale ;
- si plusieurs candidats décrivent le même concept mais des périodes,
  cohortes ou contextes différents, ne les sélectionne pas ensemble sauf si
  la question demande explicitement une comparaison ou plusieurs périodes ;
- si la question ne précise pas de période, privilégie la variante dont le
  CONTEXTE correspond à la population la plus actuelle/récente décrite par
  le schéma, plutôt que d'ajouter aussi une variante historique ;
- sélectionne plusieurs familles seulement si elles apportent des informations
  complémentaires réellement nécessaires au calcul ou à l'identification ;
- ne sélectionne pas plusieurs familles alternatives « par prudence » : si une
  seule famille contient tous les ingrédients du calcul demandé, sélectionne
  cette famille seule ;
- si la question demande d'identifier, comparer ou classer une entité/catégorie
  (par exemple « laquelle », « quelle ... », « meilleur ... »), vérifie qu'au
  moins une famille sélectionnée expose réellement l'identifiant ou le libellé
  de cette entité ; une mesure seule ne suffit pas pour nommer une entité ;
- ne considère jamais qu'un alias SQL peut créer une sémantique absente du
  schéma : row_name, row_group, school_key et year gardent leur sens réel ;
- ne choisis jamais selon l'ordre des candidats.
""".strip()

    result = llm.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "Tu sélectionnes les tables pertinentes d'une base SQLite. "
                    "Raisonne sémantiquement et retourne seulement le JSON demandé."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        schema=response_schema,
        max_completion_tokens=300,
    )

    selected_names = result.get("families", [])
    by_name = {
        candidate["table"]: candidate
        for candidate in candidates
    }

    selected = [
        by_name[name]
        for name in selected_names
        if name in by_name
    ]

    if not selected:
        raise ValueError(
            "Le LLM n'a sélectionné aucune famille valide."
        )

    return selected



def _question_needs_entity_route_validation(question):
    """
    Détecte les formulations qui demandent de classer/identifier une entité
    plutôt que de retourner seulement une mesure.

    Ce test est linguistique et générique ; aucune entité métier n'est codée.
    """
    normalized = normalize_text(question)

    patterns = (
        r"\bmeilleur(?:e|es|s)?\b",
        r"\bplus\s+(?:eleve|elevee|eleves|elevees|haut|haute|hauts|hautes|"
        r"fort|forte|forts|fortes|bas|basse|bas|basses)\b",
        r"\b(?:lequel|laquelle|lesquels|lesquelles)\b",
        r"\b(?:quel|quelle|quels|quelles)\b.{0,80}\b"
        r"(?:possede|possedent|presente|presentent|obtient|obtiennent|"
        r"atteint|atteignent|affiche|affichent)\b",
    )

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def _route_entity_candidate_summary(question, candidate, schemas):
    """
    Résumé compact centré sur la capacité d'une famille à identifier une
    catégorie/entité dans un classement.
    """
    members = [
        table
        for table in candidate.get("members", [])
        if table in schemas
    ]

    is_virtual = (
        "*" in candidate.get("table", "")
        and len(members) > 1
    )

    dimensions = []
    business_columns = []

    for table in members[:6]:
        schema = schemas.get(table, "")

        for dimension in (
            "school_key",
            "year",
            "row_name",
            "row_group",
        ):
            if re.search(
                rf"(?m)^\s*-\s+{re.escape(dimension)}\b",
                schema,
            ):
                if dimension not in dimensions:
                    dimensions.append(dimension)

        for column in parse_business_columns(schema):
            label = column.get("label", "")
            name = column.get("name", "")

            if not name:
                continue

            rendered = name
            if label and normalize_text(label) != normalize_text(name):
                rendered += f" ({label})"

            if rendered not in business_columns:
                business_columns.append(rendered)

    structure = (
        "famille virtuelle D*: chaque membre Dn est distinguable ; "
        "sur plusieurs écoles l'identité est (school_key, member_id)"
        if is_virtual
        else "famille non virtuelle"
    )

    hints = candidate_column_hints(
        question,
        candidate,
        schemas,
    )

    return (
        f"TABLE: {candidate['table']}\n"
        f"SUJET: {candidate.get('sujet', '')}\n"
        f"STRUCTURE: {structure}\n"
        f"DIMENSIONS: {', '.join(dimensions) if dimensions else '(aucune)'}\n"
        f"COLONNES MÉTIER: "
        f"{' | '.join(business_columns[:10]) if business_columns else '(aucune)'}\n"
        f"COLONNES LIÉES À LA QUESTION: "
        f"{' | '.join(hints[:8]) if hints else '(aucune)'}"
    )


def validate_or_correct_entity_route(
    llm,
    question,
    candidates,
    selected_entries,
    schemas,
):
    """
    Deuxième passe GPT uniquement pour les questions qui demandent réellement
    d'identifier/classer une entité.

    Objectif :
      - vérifier que la route permet de DISTINGUER l'entité demandée ;
      - empêcher une table qui ne distingue que les écoles de répondre à une
        question portant sur une autre catégorie ;
      - corriger le choix de familles avant toute génération SQL.

    Le modèle reste responsable du raisonnement sémantique ; Python ne connaît
    aucune notion métier particulière.
    """
    if not _question_needs_entity_route_validation(question):
        return selected_entries

    selected_names = [
        entry["table"]
        for entry in selected_entries
    ]

    candidate_blocks = [
        _route_entity_candidate_summary(
            question,
            candidate,
            schemas,
        )
        for candidate in candidates
    ]

    allowed = [
        candidate["table"]
        for candidate in candidates
    ]

    response_schema = {
        "type": "object",
        "properties": {
            "route_is_grounded": {
                "type": "boolean",
            },
            "target_entity": {
                "type": "string",
            },
            "families": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": allowed,
                },
                "minItems": 1,
                "maxItems": MAX_SELECTED_FAMILIES,
            },
        },
        "required": [
            "route_is_grounded",
            "target_entity",
            "families",
        ],
        "additionalProperties": False,
    }

    prompt = f"""
QUESTION:
{question}

ROUTE ACTUELLE:
{json.dumps(selected_names, ensure_ascii=False)}

FAMILLES DISPONIBLES:
{chr(10).join(chr(10) + block for block in candidate_blocks)}

Effectue une validation sémantique du ROUTAGE, pas du SQL.

1. Identifie l'entité/catégorie que la question demande de comparer ou de
   nommer.
2. Vérifie que les familles choisies permettent réellement de distinguer cette
   entité.
3. Si ce n'est pas le cas, corrige la route avec l'ensemble MINIMAL de familles
   disponibles permettant :
      - de calculer la mesure demandée ;
      - de distinguer l'entité demandée.
4. Une dimension school_key identifie une école uniquement. Elle ne peut pas
   être renommée pour représenter une autre entité.
5. row_name et row_group gardent leur sens réel et ne créent pas une nouvelle
   entité par alias.
6. Une famille virtuelle D* peut distinguer ses membres Dn. Sur plusieurs
   établissements, un membre doit être identifié par le couple
   (school_key, member_id), jamais par Dn seul.
7. Une famille descriptive complémentaire peut être sélectionnée si elle est
   réellement nécessaire pour NOMMER l'entité, mais ne l'ajoute pas si la
   famille de mesure permet déjà de conserver une identité structurelle fiable.
8. Ne choisis pas une table qui permet seulement de classer les écoles lorsque
   la question demande une autre catégorie.

Retourne route_is_grounded=true si la route finale permet réellement de
répondre à la question, ainsi que cette route finale dans families.
""".strip()

    result = llm.chat_json(
        [
            {
                "role": "system",
                "content": (
                    "Tu valides le grounding sémantique d'un routage de tables "
                    "SQLite. Tu corriges seulement la sélection de familles."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        schema=response_schema,
        max_completion_tokens=500,
    )

    selected_names = result.get("families", [])
    by_name = {
        candidate["table"]: candidate
        for candidate in candidates
    }

    corrected = [
        by_name[name]
        for name in selected_names
        if name in by_name
    ]

    if not corrected:
        return selected_entries

    return corrected



def expand_families(selected_entries):
    tables = []

    for entry in selected_entries:
        for table in entry.get("members", []):
            if table not in tables:
                tables.append(table)

    return tables


# ---------------------------------------------------------------------------
# Prompt SQL : court, générique et guidé par le semantic_schema
# ---------------------------------------------------------------------------

def build_sql_context(selected_entries, schemas):
    return "\n\n".join(
        compact_family_schema(entry, schemas)
        for entry in selected_entries
    )


def build_sql_prompt(question, selected_entries, schemas, data_profile=''):
    schema_context = build_sql_context(
        selected_entries,
        schemas,
    )

    schools = ", ".join(POLYTECH_SCHOOLS)

    prompt = f"""
QUESTION:
{question}

SCHÉMA SÉMANTIQUE AUTORISÉ:
{schema_context}

SCHOOL_KEY DU RÉSEAU POLYTECH:
{schools}

PROFIL DE DISPONIBILITÉ DES DONNÉES DANS LE PÉRIMÈTRE DEMANDÉ:
{data_profile or "Non fourni."}

Ce profil indique seulement quelles colonnes du schéma sont réellement
renseignées dans le périmètre de la question. Il ne définit pas la formule.

Génère une seule requête SQLite SELECT/WITH qui répond exactement
à la question.

Règles générales:
- utilise uniquement les tables, colonnes et valeurs décrites ci-dessus ;
- fais confiance au SUJET, au CONTEXTE, aux libellés, aux types, unités
  et règles d'agrégation du schéma pour comprendre les données ;
- n'invente ni table, ni colonne, ni catégorie ;
- si une école est demandée, filtre son school_key ;
- lorsqu'une question demande combien d'entités POSSÈDENT, ONT ou SATISFONT une
  propriété booléenne/catégorielle, IS NOT NULL, TRIM(...) <> '' ou
  NULLIF(TRIM(...), '') IS NOT NULL ne suffisent JAMAIS : ces tests vérifient
  seulement qu'une réponse existe, pas qu'elle est affirmative ;
- avant de compter, interprète la valeur textuelle elle-même : une réponse
  explicitement négative ou exprimant l'absence de la propriété (par exemple
  « non », « aucun », « aucune », « néant », « sans ... », « pas de ... »)
  doit être EXCLUE du compteur. Une valeur non vide n'est positive que si son
  contenu apporte réellement une preuve affirmative de la propriété demandée ;
- pour un champ texte libre servant à renseigner le nom d'un label, dispositif,
  certification ou autre propriété, compte une entité seulement si le contenu
  renseigne effectivement un élément positif ; ne compte jamais une simple
  réponse négative sous prétexte qu'elle est non NULL ;
- si la question vise une propriété nommée précisément, filtre le CONTENU de
  la colonne sur cette propriété, et pas seulement sur l'existence d'un texte
  affirmatif. Cette règle vaut pour lister comme pour compter les entités.
  Le libellé d'une rubrique peut citer plusieurs exemples : il ne prouve pas
  que chaque réponse possède celui demandé. Un autre label/certification ou
  dispositif positif n'est pas équivalent à celui demandé. Tiens compte des
  variantes typographiques du même nom sans élargir à des notions voisines,
  et ne confonds pas un objectif de candidature avec une attribution acquise ;
- convention de cette application : les demandes générales de labels de
  développement durable, y compris « quelles écoles ont un label DDRS »,
  portent sur l'ensemble des déclarations positives de la rubrique cti_v_5.
  DD&RS, Eco-campus et les autres dispositifs déclarés dans cette rubrique
  sont inclus sans liste fermée de noms ni filtre LIKE '%ddrs%' ou égalité
  sur 'Label DD&RS'. Retourne le texte déclaré avec l'école, sans restreindre
  aux seules déclarations contenant le sigle DDRS. Exclue les réponses vides,
  négatives ou exprimant seulement un projet/candidature. Ne limite à un label
  particulier que si la question demande explicitement ce label à l'exclusion
  des autres. Cette convention précise le sens de la propriété demandée ;
- si aucune école précise n'est demandée, le périmètre est EXACTEMENT le
  réseau Polytech listé ci-dessus : filtre explicitement school_key avec ces
  valeurs ; ne laisse jamais la requête agréger toutes les écoles présentes
  physiquement dans la base ;
- pour toute union de tables ou de voies, construis OBLIGATOIREMENT une CTE
  commune de périmètre APRÈS l'union complète, en filtrant school_key sur les
  écoles demandées (les 16 ci-dessus pour une question générale). Conserve
  school_key dans chaque branche pour permettre ce filtre. Des filtres présents
  seulement dans certaines branches, même D1/D2, ne remplacent pas cette CTE ;
- calcule la campagne, les numérateurs, dénominateurs, taux et écoles
  contributrices exclusivement depuis cette CTE filtrée ou ses descendants.
  Après le filtre commun, ne relis aucune branche ou table non filtrée pour
  MAX(year), SUM, COUNT ou une jointure qui réintroduirait des observations ;
- vérifie avant de retourner le SQL que chaque chemin entre une table source
  et le calcul final traverse ce filtre commun. Pour une question générale,
  COUNT(DISTINCT school_key) est au plus 16. N'utilise jamais LIMIT, MIN(...,16)
  ni un plafond artificiel pour masquer un dépassement : corrige le périmètre ;
- si une année est explicitement demandée, filtre EXACTEMENT cette année ;
  ne la remplace jamais par MAX(year) ni par la dernière campagne disponible ;
- si aucune année n'est demandée, utilise la campagne pertinente la plus
  récente où les données nécessaires à la réponse sont exploitables ;
- même lorsqu'aucune année n'est demandée, ne mélange jamais plusieurs
  campagnes dans une même agrégation : sélectionne explicitement une seule
  campagne récente avant le calcul ;
- pour une agrégation multi-écoles ou multi-tables, utilise une même campagne ;
- une famille contenant * est virtuelle : ses membres physiques sont des
  instances parallèles du même indicateur logique ;
- si la question porte sur l'ensemble d'une école ou du réseau et ne désigne
  pas explicitement un membre/sous-ensemble de cette famille, utilise TOUS les
  membres physiques de la famille ; ne prends jamais le premier membre comme
  représentant de l'ensemble ;
- pour une famille virtuelle agrégée, construis d'abord l'ensemble des membres
  (par exemple via UNION ALL), puis détermine la campagne pertinente sur cet
  ensemble ; ne déduis pas la campagne à partir d'un seul membre représentatif ;
- pour une mesure dérivée, déduis numérateur, dénominateur ou poids à partir
  de la sémantique des colonnes ; prends toutes les composantes pertinentes
  sauf si la question en limite explicitement le périmètre ;
- si plusieurs composantes parallèles sémantiquement compatibles sont
  renseignées dans le profil de disponibilité et que la question est générale,
  ne choisis pas arbitrairement une seule composante : intègre toutes celles
  qui appartiennent réellement à la population demandée ;
- pour une synthèse générale d'apprenants ou de diplômés, inclure FISE ET
  FISA + FISEA sur CHAQUE membre physique pertinent, sauf restriction explicite
  de la question. Vérifie la matrice complète membres × voies avant de rendre
  le SQL : une UNION comprenant FISA/FISEA pour tous les Dn mais FISE seulement
  pour D1 est incomplète. Chaque couple membre/voie compatible doit être lu ;
- pour les taux, ratios ou moyennes pondérées, normalise ces contributions
  séparément avec une constante source_table contenant le vrai nom physique,
  une constante voie et le couple mesure/réponses ou valeur/poids de cette voie.
  Ne remplace jamais source_table par 'fise' : ce libellé appartient à voie.
  Déduplique avec les deux identités, puis valide chaque couple séparément,
  avant d'agréger ensemble les contributions FISE et FISA/FISEA. Une voie vide
  ne doit ni devenir un zéro observé ni empêcher une autre voie exploitable de
  contribuer. La couverture de 16 écoles ne prouve pas que toutes les voies ou
  spécialités ont été incluses. Ne filtre pas simplement les observations
  incohérentes pour embellir le taux ; conserve leur signalement ;
- pour des colonnes de type count organisées en populations parallèles dans
  leurs libellés, une question générale doit additionner toutes les populations
  réellement renseignées ; si la question limite explicitement une population,
  respecte cette restriction ;
- lorsque plusieurs composantes count parallèles sont additionnées ligne à
  ligne, traite une composante NULL comme absente avec COALESCE(composante, 0)
  au lieu de laisser l'addition entière devenir NULL. Cette addition concerne
  les effectifs indépendants ; pour un ratio ou une pondération, construis et
  valide d'abord chaque couple mesure/dénominateur par voie, sans remplacer les
  valeurs manquantes du couple par zéro ;
- conserve un sous-groupe cohérent entre tables et populations : ne mélange
  jamais une colonne « Total » avec une colonne d'un sous-groupe différent
  simplement parce qu'une table ne possède pas la colonne Total ;
- un alias SQL ne change jamais le sens d'une dimension. N'écris pas par
  exemple row_name AS <concept demandé> si les valeurs observées de row_name
  ne représentent pas réellement ce concept ;
- si une colonne métier dédiée au concept demandé existe dans une famille
  sélectionnée, utilise cette colonne plutôt que de renommer row_name/row_group
  avec le même nom ;
- si la question demande « quel/quelle X », un classement ou un maximum par X,
  le résultat doit contenir un véritable identifiant/libellé de X provenant du
  schéma. Ne remplace pas X par school_key, row_name ou row_group si ces
  dimensions ne représentent pas explicitement X ;
- pour une famille virtuelle D* qui représente plusieurs membres parallèles,
  si la question demande de comparer/identifier une spécialité ou filière,
  conserve explicitement l'identité du membre dans le SQL (par exemple
  'd3' AS member_id pour la table ..._d3_...) ;
- IMPORTANT : un identifiant d1/d2/d3... n'a de sens qu'à l'intérieur d'une
  école. Si aucune école précise n'est demandée, l'identité d'une formation est
  obligatoirement le couple (school_key, member_id) : groupe, compare et
  retourne les deux. Ne fusionne jamais les d3 de plusieurs écoles comme s'ils
  représentaient la même spécialité ;
- lorsque plusieurs familles distinctes sont combinées et représentent des
  catégories différentes, conserve dans le résultat une colonne permettant de
  distinguer leur provenance au lieu de perdre cette information dans UNION ALL ;
- le mot « minimum » ou « maximum » peut faire partie de la définition d'un
  indicateur. Ne fais pas MIN()/MAX() entre des populations distinctes seulement
  parce que le SUJET de chaque famille contient ce mot ; si les familles
  décrivent des populations différentes, conserve ces catégories sauf si la
  question demande réellement de les comparer entre elles ;
- respecte strictement les indications d'agrégation du schéma ;
- sans école ou ventilation explicitement demandée, produis une synthèse globale
  du réseau : pas de GROUP BY school_key dans le résultat final. Pour un ratio de
  catégories, somme leurs effectifs compatibles avant la division ; pour une
  proportion, calcule 100.0 * SUM(effectif_categorie) / SUM(effectif_population).
  Indique précisément la population retenue. Si seuls des pourcentages sans poids
  sont disponibles, expose la limite ; une liste par école n'est pas une réponse
  complète à une demande globale ;
- conserve l'horizon de la question quand il est explicite. Sinon, n'invente pas
  une contrainte temporelle spécifique à la source sélectionnée : utilise la
  campagne exploitable récente et le contexte réel de la mesure, en retournant
  un intitulé fidèle à ce contexte. Une observation en janvier ne doit pas être
  renommée « à six mois » ; un effectif en recherche d'emploi ne doit pas devenir
  un effectif en emploi par simple complément sans tous les autres statuts ;
- une colonne de type percentage ou d'unité percent contient des parts sur 100,
  pas des effectifs : ne somme jamais ces valeurs entre écoles/populations et
  ne les présente jamais comme des personnes. Pour un ratio entre catégories,
  utilise SUM(effectif_categorie_A) / NULLIF(SUM(effectif_categorie_B), 0).
  À défaut d'effectifs directs, les parts peuvent être pondérées uniquement par
  les effectifs totaux correspondants, à école, population et période identiques.
  Une moyenne simple de parts ou un quotient de sommes de pourcentages donne le
  même poids à chaque école et ne mesure pas le ratio global des personnes.
  Sans effectifs/poids compatibles dans les familles disponibles, retourne les
  pourcentages par école pour exposer la limite, sans inventer de ratio global.
  Respecte not_aggregatable : des unités hétérogènes exigent une clarification ;
- pour un chiffre de réseau, fixe d'abord le grain des observations : école,
  campagne, promotion/horizon, membre, voie et catégorie lorsque pertinents.
  Déduplique les mêmes observations répétées dans plusieurs documents avant
  de sommer, y compris pour les effectifs et numérateurs/dénominateurs de taux.
  Conserve ces identités : deux spécialités ayant le même effectif ne sont pas
  un doublon. N'utilise jamais SUM(DISTINCT valeur) pour dédupliquer des entités ;
- un total et ses sous-totaux se recouvrent : utilise soit Total soit les
  composantes disjointes, jamais Total + Femmes + Hommes ni un effectif et
  ses colonnes « dont ». Une catégorie manquante n'est pas un zéro observé ;
- pour un taux de réseau calculable à partir d'effectifs, utilise
  100.0 * SUM(numerateur) / NULLIF(SUM(denominateur), 0), pas AVG(taux_ecole).
  Numérateur et dénominateur doivent porter sur les mêmes écoles, membres,
  voies, catégories, promotion et horizon ; conserve uniquement les couples
  numérateur renseigné et dénominateur strictement positif. Un numérateur zéro
  est valide. Ne rattache pas à un numérateur manquant les réponses d'une autre
  voie, question ou promotion, et ne répète pas le dénominateur dans une jointure ;
- avant de finaliser un ratio, vérifie le chemin complet des lignes dans les
  CTE, WHERE, HAVING et jointures : aucune étape ne doit supprimer une catégorie
  nécessaire au numérateur ou au dénominateur avant leur mise en correspondance.
  Si les effectifs sont sur des lignes distinctes, conserve l'union de leurs
  catégories dans les filtres, puis pivote par école/campagne/membre/voie/horizon
  avec MAX(CASE WHEN ... THEN valeur END) après déduplication, sans ELSE 0.
  Exemple structurel : WHERE row_name IN ('catégorie numérateur',
  'catégorie dénominateur'), puis appariement des deux valeurs au même grain.
  Ces deux libellés illustratifs doivent être remplacés par les valeurs observées
  du schéma. N'utilise pas COALESCE(valeur, 0) pour inventer un effectif manquant.
  Filtre les couples numerateur IS NOT NULL AND denominateur > 0 avant les
  sommes et le COUNT(DISTINCT school_key). Pour plusieurs horizons, construis
  et valide les couples séparément, avec une couverture propre à chaque taux ;
- MAX(CASE...) ne sert à pivoter qu'une catégorie unique déjà dédupliquée.
  Si le numérateur regroupe plusieurs catégories disjointes, somme leurs
  effectifs après déduplication, par exemple SUM(CASE WHEN catégorie IN (...)
  THEN effectif END), plutôt que MAX qui ne garderait que la plus grande tranche.
  Ne cumule pas un total avec ses composantes. Conserve la distinction entre
  absence d'une catégorie et zéro observé ;
- pour une enquête, distingue part parmi les répondants, part parmi tous les
  diplômés et taux net d'emploi parmi les actifs. Utilise le dénominateur
  correspondant à la définition demandée et retourne de quoi le justifier.
  Sans définition explicite, indique dans les alias le dénominateur réellement
  disponible ; ne présente pas une part parmi les répondants comme une mesure
  de tous les diplômés. Ne déduis pas l'emploi de 100 moins le taux de recherche
  sans vérifier les autres statuts et l'identité des populations ;
- lorsqu'une mesure possède son propre nombre de réponses (response_count),
  utilise par défaut ce nombre de réponses pour sa proportion parmi les
  répondants au champ, sauf dénominateur différent explicitement demandé.
  Apparier la mesure et ses réponses à école, campagne, membre, voie et
  sous-groupe identiques ; conserver uniquement les couples mesure non NULL
  et réponses > 0, dédupliquer puis calculer 100.0 * SUM(mesure) / SUM(réponses).
  Ne remplace pas les réponses au champ par un effectif d'une autre rubrique,
  même si celle-ci décrit une population apparemment proche. Les restrictions
  de lieu, contrat, statut et exclusions du champ s'appliquent à la population
  mesurée. Si les réponses locales manquent, expose la limite sans substituer
  silencieusement un autre dénominateur. N'utilise pas toutes les réponses à
  tous les champs ou tous les sous-groupes pour le poids d'une seule mesure ;
- ne confonds pas un emploi trouvé en moins de deux mois, une durée de recherche
  d'au moins six mois et une recherche d'emploi toujours en cours à six mois.
  Utilise les tranches exactes observées. Ne crée pas une borne absente ; pour
  une tranche seulement approchante, conserve son libellé exact dans le résultat.
  Distingue les emplois obtenus avant le diplôme de ceux obtenus après le diplôme ;
- une demande temporelle ambiguë ne permet pas de renommer une observation
  à 6 mois ou 12–15 mois « à la sortie ». Conserve l'horizon exact de la colonne
  dans les alias et retourne de quoi expliquer cette limite ;
- pour les apprentis, distingue inscrits FISA, étudiants FISE en contrat
  d'apprentissage en dernière année, diplômés FISA, bachelors, spécialisations
  et contrats de professionnalisation. Ne les additionne que si le périmètre
  demandé les inclut et si les populations sont disjointes. Un nombre de
  créations d'entreprise ou d'étudiants entrepreneurs ne mesure pas le nombre
  de diplômés sous régime juridique autoentrepreneur ;
- dans une table comportant plusieurs horizons, un nom générique Total ou
  Total_2 ne suffit pas à identifier une promotion. Utilise une correspondance
  sémantique établie, pas l'ordre du suffixe. Si les métadonnées sont ambiguës,
  ne devine pas et retourne les données permettant d'expliquer la limite ;
- si SUJET/CONTEXTE et type/unité/agrégation se contredisent, ne produis pas
  une somme absurde (par exemple additionner des salaires étiquetés count).
  Une colonne existante ne prouve pas une extraction correcte. Ne remplace pas
  une mesure absente par un indicateur voisin ni un numérateur par un total ;
- retourne, avec les synthèses du réseau, la campagne, le nombre d'écoles
  contributrices et, pour les taux, le numérateur et le dénominateur utilisés.
  La présence d'une école dans la base ne garantit pas sa contribution à toutes
  les mesures. Une absence de données ne prouve pas une absence de la propriété ;
- pour une couverture globale, COUNT(DISTINCT school_key) doit être calculé
  sur l'ensemble des observations valides, sans GROUP BY school_key. Dans un
  résultat ventilé par école, ce compteur vaut mécaniquement 1 par ligne :
  nomme-le comme tel ou fournis séparément la couverture globale. N'utilise pas
  un compteur local comme couverture du réseau. Si un résultat diagnostique
  présente des parts par école, conserve school_key dans les colonnes finales ;
- pour une mesure marquée weighted_average_by_response_count, calcule une
  moyenne pondérée :
      1.0 * SUM(valeur * poids) / NULLIF(SUM(poids), 0)
  où le poids est la colonne de type response_count correspondant à la même
  population, au même sous-groupe (par exemple F/H/Total) et à la même voie ;
- pour une question générale sur le salaire médian d'une école ou du réseau,
  retourne une synthèse, pas la liste des salaires de chaque spécialité/voie.
  Combine les couples salaire/Nombre de réponses Total des voies pertinentes
  (ou le sous-groupe explicitement demandé) et de tous les membres compatibles
  dans un même calcul. Ne cumule pas
  Total avec F et H. Sauf demande de détail/comparaison, ne groupe pas par
  school_key, member_id ou voie : une seule valeur par year et catégorie
  row_name (Avec prime / Sans prime). Conserve row_group s'il distingue
  réellement des catégories. N'arrondis qu'après le calcul ;
- avant une synthèse pondérée, déduplique les observations identiques répétées
  dans plusieurs documents, sans fusionner des membres ou populations distincts.
  Dans chaque branche UNION ALL, conserve une constante source_table (nom de
  la table physique) et une constante voie identifiant la population et le couple
  valeur/poids. Utilise ensuite une CTE SELECT DISTINCT sur source_table,
  school_key, year, row_name, row_group, voie, valeur et poids.
  Deux membres ou deux voies ayant les mêmes chiffres restent deux contributions.
  Ne remplace pas simplement UNION ALL par UNION sur des branches dépourvues
  de ces identifiants. Ne conserve pas les identifiants techniques de document
  dans la clé de déduplication d'une même observation ;
- choisis la dernière campagne sur cette CTE après restriction aux écoles
  demandées et aux couples valeur non NULL / poids > 0, et non sur les lignes
  brutes éventuellement vides. Agrège ensuite les observations dédupliquées ;
- pour la synthèse des salaires, FISE et FISA + FISEA contribuent au même
  numérateur et au même dénominateur dans chaque catégorie de primes.
  La voie reste dans la clé de déduplication, mais pas dans le GROUP BY final,
  sauf demande explicite de détail par voie. Nomme le résultat
  moyenne_ponderee_des_medianes et arrondis seulement la valeur finale ;
- applique toujours le filtre school_key dans la CTE commune après UNION ALL,
  AVANT de choisir la campagne ou de calculer ; les filtres de branche
  éventuels sont seulement complémentaires ;
- pour cette moyenne pondérée, utilise uniquement les lignes où la valeur est
  non NULL et le poids est strictement positif ;
- n'utilise jamais AVG(valeur), MIN(valeur), MAX(valeur) ou SUM(valeur) seuls
  pour synthétiser une mesure weighted_average_by_response_count ;
- si plusieurs voies/populations parallèles pertinentes sont renseignées et
  que la question est générale, inclus tous les couples valeur/poids
  correspondants au lieu d'en choisir arbitrairement un seul ;
- si plusieurs valeurs de row_name/row_group correspondent à la mesure et que
  la question n'en choisit aucune, ne les fusionne pas arbitrairement : retourne
  la dimension avec les valeurs métier pertinentes ;
- utilise row_name/row_group seulement si leur valeur exacte est nécessaire ;
  pour lister les catégories, sélectionne la dimension au lieu d'inventer
  un filtre ;
- si tu déduis l'année, retourne-la dans le résultat ;
- préfère un SQL compact avec CTE et UNION ALL lorsqu'il y a beaucoup de
  tables homologues ;
- si le schéma ne permet pas un calcul demandé de façon correcte, retourne
  les données les plus directement pertinentes plutôt que d'inventer
  une formule.
""".strip()

    return prompt, schema_context


def clean_sql(text):
    return (
        (text or "")
        .replace("```sql", "")
        .replace("```SQL", "")
        .replace("```", "")
        .strip()
    )


def generate_sql(llm, question, selected_entries, schemas, data_profile=''):
    prompt, schema_context = build_sql_prompt(
        question,
        selected_entries,
        schemas,
        data_profile=data_profile,
    )

    sql = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Tu es expert SQLite et analyse de données. "
                    "Le schéma sémantique est la source de vérité. "
                    "Retourne uniquement la requête SQL, sans markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_completion_tokens=3000,
    )

    return clean_sql(sql), schema_context


# ---------------------------------------------------------------------------
# Validation générique Python
# ---------------------------------------------------------------------------

def referenced_physical_tables(sql):
    """
    Les vraies tables de ce projet commencent par cti_ ou defi_.
    Les noms de CTE ne sont donc pas confondus avec elles.
    """
    return set(
        re.findall(
            r"\b(?:cti|defi)_[a-zA-Z0-9_]+\b",
            sql,
            flags=re.IGNORECASE,
        )
    )



def _physical_select_branches(sql, physical_tables):
    """
    Extrait les SELECT qui lisent directement une table physique autorisée.
    Le but est de valider les colonnes branche par branche avant SQLite.
    """
    if not physical_tables:
        return []

    table_pattern = "|".join(
        sorted(
            (
                re.escape(table)
                for table in physical_tables
            ),
            key=len,
            reverse=True,
        )
    )

    pattern = re.compile(
        rf"""
        \bSELECT\b
        (?P<select>.*?)
        \bFROM\s+
        (?P<table>{table_pattern})\b
        (?:\s+(?:AS\s+)?(?P<alias>[a-zA-Z_][a-zA-Z0-9_]*))?
        """,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

    return [
        {
            "table": match.group("table"),
            "alias": match.group("alias"),
            "select": match.group("select"),
        }
        for match in pattern.finditer(sql)
    ]


def validate_columns_per_physical_table(
    connection,
    sql,
    physical_tables,
):
    """Vérifie les colonnes physiques en respectant les alias SQL."""
    table_columns = {
        table.lower(): set(schema_columns_from_sqlite(connection, table))
        for table in physical_tables
    }

    alias_to_table = {}

    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+((?:cti|defi)_[a-zA-Z0-9_]+)"
        r"(?:\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*))?",
        sql,
        flags=re.IGNORECASE,
    ):
        table = match.group(1).lower()
        alias = match.group(2)

        if table not in table_columns:
            continue

        alias_to_table[table] = table
        if alias:
            alias_to_table[alias.lower()] = table

    problems = []

    for match in re.finditer(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b",
        sql,
    ):
        alias = match.group(1).lower()
        column = match.group(2).lower()
        table = alias_to_table.get(alias)

        if table and column not in table_columns.get(table, set()):
            problems.append((table, column))

    if not problems:
        return True, None

    grouped = {}
    for table, column in problems:
        grouped.setdefault(table, set()).add(column)

    lines = [
        "Certaines colonnes qualifiées n'existent pas dans leur table physique :"
    ]

    for table, missing in sorted(grouped.items()):
        lines.append(
            f"- {table}: colonnes absentes = {', '.join(sorted(missing))}"
        )

    lines.append(
        "Corrige chaque référence en respectant l'alias de sa table."
    )

    return False, "\n".join(lines)


def allowed_dimension_values(physical_tables, schemas, dimension):
    values = []

    for table in physical_tables:
        schema = schemas.get(table, "")

        for value in get_dimension_values(schema, dimension):
            if value not in values:
                values.append(value)

    return values


def closest_observed_value(value, observed, threshold=0.94, margin=0.03):
    """
    Résout seulement les écarts de forme très proches entre une valeur produite
    par le LLM et une valeur réellement observée dans le schéma.

    Exemple générique :
        "Total des répondants" -> "Total de répondants"

    On ne corrige rien si le rapprochement n'est pas suffisamment sûr.
    """
    if not observed:
        return None

    normalized_value = normalize_text(value).strip()

    exact = {
        normalize_text(candidate).strip(): candidate
        for candidate in observed
    }

    if normalized_value in exact:
        return exact[normalized_value]

    scored = sorted(
        (
            SequenceMatcher(
                None,
                normalized_value,
                normalize_text(candidate).strip(),
            ).ratio(),
            candidate,
        )
        for candidate in observed
    )

    best_score, best_value = scored[-1]
    second_score = scored[-2][0] if len(scored) > 1 else 0.0

    if (
        best_score >= threshold
        and best_score - second_score >= margin
    ):
        return best_value

    return None


def canonicalize_dimension_literals(sql, physical_tables, schemas):
    """
    Remplace uniquement les valeurs row_name / row_group presque identiques
    à une valeur observée par leur forme canonique du semantic_schema.

    C'est une normalisation structurelle, pas une règle métier.
    """
    result = sql

    for dimension in ("row_name", "row_group"):
        observed = allowed_dimension_values(
            physical_tables,
            schemas,
            dimension,
        )

        if not observed:
            continue

        # dimension = 'valeur'
        equality_pattern = re.compile(
            rf"((?:\b[a-zA-Z_][a-zA-Z0-9_]*\.)?"
            rf"\b{dimension}\b\s*=\s*)'((?:''|[^'])*)'",
            flags=re.IGNORECASE,
        )

        def replace_equality(match):
            original = match.group(2).replace("''", "'")
            canonical = closest_observed_value(
                original,
                observed,
            )

            if canonical is None:
                return match.group(0)

            escaped = canonical.replace("'", "''")
            return f"{match.group(1)}'{escaped}'"

        result = equality_pattern.sub(
            replace_equality,
            result,
        )

        # dimension IN ('a', 'b', ...)
        in_pattern = re.compile(
            rf"((?:\b[a-zA-Z_][a-zA-Z0-9_]*\.)?"
            rf"\b{dimension}\b\s+IN\s*\()((?:'(?:''|[^'])*'|[^')])*)(\))",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def replace_in(match):
            body = match.group(2)

            def replace_literal(literal_match):
                original = literal_match.group(1).replace("''", "'")
                canonical = closest_observed_value(
                    original,
                    observed,
                )

                if canonical is None:
                    return literal_match.group(0)

                escaped = canonical.replace("'", "''")
                return f"'{escaped}'"

            body = re.sub(
                r"'((?:''|[^'])*)'",
                replace_literal,
                body,
            )

            return match.group(1) + body + match.group(3)

        result = in_pattern.sub(
            replace_in,
            result,
        )

    return result


def explicit_years_in_question(question):
    """Extrait les années calendaires explicitement mentionnées."""
    years = []
    for value in re.findall(r"\b(?:19|20)\d{2}\b", question):
        year = int(value)
        if year not in years:
            years.append(year)
    return years


def validate_explicit_year_scope(question, sql):
    """
    Une année explicitement demandée est une contrainte utilisateur, pas une
    préférence. Pour une année unique, le SQL doit réellement filtrer dessus.
    """
    years = explicit_years_in_question(question)

    if not years:
        return True, None

    # Cas courant : une année précise demandée.
    if len(years) == 1:
        year = years[0]
        year_ref = r"(?:\b[a-zA-Z_][a-zA-Z0-9_]*\.)?\byear\b"
        patterns = [
            rf"{year_ref}\s*=\s*{year}\b",
            rf"\b{year}\s*=\s*{year_ref}",
            rf"{year_ref}\s+IN\s*\([^)]*\b{year}\b[^)]*\)",
        ]

        if not any(re.search(p, sql, flags=re.IGNORECASE | re.DOTALL) for p in patterns):
            return False, (
                f"La question demande explicitement l'année {year}, mais le SQL "
                f"ne filtre pas year sur {year}. N'utilise pas MAX(year) à la place."
            )

        return True, None

    # Plusieurs années : au minimum, elles doivent toutes être présentes dans
    # l'expression SQL afin que le correcteur ne les ignore pas silencieusement.
    missing = [year for year in years if not re.search(rf"\b{year}\b", sql)]
    if missing:
        return False, (
            "La question mentionne explicitement plusieurs années mais le SQL "
            "en ignore certaines : " + ", ".join(map(str, missing))
        )

    return True, None


def validate_aggregation_contracts(sql, physical_tables, schemas):
    """
    Fait respecter les contrats d'agrégation portés par le semantic_schema.

    - do_not_average_medians : compatibilité avec d'anciens schémas ;
    - weighted_average_by_response_count : autorise une somme pondérée mais
      rejette les agrégations naïves d'une statistique déjà agrégée.
    """
    forbidden_direct = set()
    weighted = set()

    for table in physical_tables:
        schema = schemas.get(table, "")

        for column in parse_business_columns(schema):
            aggregation = column.get("aggregation")
            name = column["name"].lower()

            if aggregation == "do_not_average_medians":
                forbidden_direct.add(name)

            elif aggregation == "weighted_average_by_response_count":
                weighted.add(name)

    all_special = forbidden_direct | weighted

    referenced = {
        name
        for name in all_special
        if re.search(
            rf"\b{re.escape(name)}\b",
            sql,
            flags=re.IGNORECASE,
        )
    }

    if not referenced:
        return True, None

    # Suivre les alias simples :
    # fise_salaire_median_total AS valeur
    alias_to_source = {}

    for name in referenced:
        for match in re.finditer(
            rf"\b{re.escape(name)}\b\s+AS\s+"
            rf"([a-zA-Z_][a-zA-Z0-9_]*)",
            sql,
            flags=re.IGNORECASE,
        ):
            alias_to_source[match.group(1).lower()] = name

    names_to_check = set(referenced) | set(alias_to_source)

    for name in names_to_check:
        source = alias_to_source.get(name, name)

        if source in forbidden_direct:
            aggregate = re.search(
                rf"\b(AVG|SUM|MIN|MAX)\s*\(\s*"
                rf"(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?"
                rf"{re.escape(name)}\b",
                sql,
                flags=re.IGNORECASE,
            )

            if aggregate:
                return False, (
                    f"La mesure '{source}' est marquée "
                    "do_not_average_medians. "
                    f"{aggregate.group(1).upper()} ne peut pas être appliqué "
                    "à plusieurs valeurs déjà agrégées."
                )

        if source in weighted:
            # AVG/MIN/MAX sont toujours incorrects pour synthétiser la mesure.
            direct_bad = re.search(
                rf"\b(AVG|MIN|MAX)\s*\(\s*"
                rf"(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?"
                rf"{re.escape(name)}\b",
                sql,
                flags=re.IGNORECASE,
            )

            if direct_bad:
                return False, (
                    f"La mesure '{source}' doit être agrégée selon "
                    "weighted_average_by_response_count. "
                    f"{direct_bad.group(1).upper()}({name}) est une "
                    "agrégation non pondérée. Utilise "
                    "SUM(valeur * response_count) / SUM(response_count)."
                )

            # SUM(valeur) seul est incorrect, mais SUM(valeur * poids) est permis.
            plain_sum = re.search(
                rf"\bSUM\s*\(\s*"
                rf"(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?"
                rf"{re.escape(name)}\s*\)",
                sql,
                flags=re.IGNORECASE,
            )

            if plain_sum:
                return False, (
                    f"La mesure '{source}' doit être agrégée selon "
                    "weighted_average_by_response_count. "
                    f"SUM({name}) seul est interdit. Utilise "
                    "SUM(valeur * response_count) / SUM(response_count)."
                )

    return True, None


def _label_parts(label):
    return [
        part.strip()
        for part in (label or "").split(">")
        if part.strip()
    ]


def _semantic_column_parts(column):
    """
    Décompose un libellé hiérarchique générique en :
      population > mesure > sous-groupe

    Exemples de formes supportées :
      A > Nombre
      A > Nombre > Total
      Nombre total

    Les noms métier concrets ne sont jamais codés ici.
    """
    parts = _label_parts(column.get("label", ""))

    if len(parts) >= 3:
        population = " > ".join(parts[:-2])
        measure = parts[-2]
        subgroup = parts[-1]
    elif len(parts) == 2:
        population = parts[0]
        measure = parts[1]
        subgroup = ""
    elif len(parts) == 1:
        population = ""
        measure = parts[0]
        subgroup = ""
    else:
        population = ""
        measure = column.get("name", "")
        subgroup = ""

    return {
        "name": column.get("name", "").lower(),
        "type": column.get("type", ""),
        "aggregation": column.get("aggregation", ""),
        "population": population,
        "measure": measure,
        "subgroup": subgroup,
        "label": column.get("label", ""),
    }


def _semantic_column_specs(physical_tables, schemas):
    """
    Fusionne les métadonnées de colonnes de tous les membres d'une famille.
    Une même colonne peut manquer dans certains membres physiques.
    """
    specs = {}

    for table in physical_tables:
        for column in parse_business_columns(schemas.get(table, "")):
            name = column.get("name", "").lower()
            if not name:
                continue

            spec = _semantic_column_parts(column)
            existing = specs.get(name)

            # Garde de préférence la version qui possède le plus de sémantique.
            if (
                existing is None
                or len(spec.get("label", "")) > len(existing.get("label", ""))
            ):
                specs[name] = spec

    # Associer génériquement les response_count à une population/sous-groupe.
    response_by_scope = {}

    for spec in specs.values():
        if spec["type"] != "response_count":
            continue

        key = (
            normalize_text(spec["population"]),
            normalize_text(spec["subgroup"]),
        )
        response_by_scope.setdefault(key, []).append(spec["name"])

    for spec in specs.values():
        key = (
            normalize_text(spec["population"]),
            normalize_text(spec["subgroup"]),
        )
        weights = response_by_scope.get(key, [])
        spec["response_count"] = weights[0] if weights else None

    return specs


def _sql_references_name(sql, name):
    return bool(
        re.search(
            rf"\b{re.escape(name)}\b",
            sql,
            flags=re.IGNORECASE,
        )
    )


def _question_mentions_population(question, population):
    """
    Détection purement lexicale à partir du libellé du schéma.
    Aucun synonyme métier n'est codé.
    """
    population_tokens = {
        token
        for token in text_tokens(population)
        if len(token) >= 3
    }

    if not population_tokens:
        return False

    question_tokens = set(text_tokens(question))
    return bool(population_tokens & question_tokens)


def _validation_scope(connection, question, year):
    requested_school = question_requested_school(question)

    clauses = ["year = ?"]
    params = [year]

    if requested_school:
        clauses.append("school_key = ?")
        params.append(requested_school)
    else:
        placeholders = ", ".join("?" for _ in POLYTECH_SCHOOLS)
        clauses.append(f"school_key IN ({placeholders})")
        params.extend(POLYTECH_SCHOOLS)

    return " AND ".join(clauses), params


def _infer_validation_year(
    connection,
    question,
    physical_tables,
    metric_names,
    result_columns=None,
    result_rows=None,
):
    years = explicit_years_in_question(question)

    if len(years) == 1:
        return years[0]

    # Préférer l'année effectivement retournée par le SQL.
    if result_columns and result_rows:
        lowered = [str(column).lower() for column in result_columns]

        if "year" in lowered:
            index = lowered.index("year")
            result_years = {
                int(row[index])
                for row in result_rows
                if row[index] is not None
            }

            if len(result_years) == 1:
                return next(iter(result_years))

    # Si le SQL n'a pas renvoyé year, déduire la campagne récente où au moins
    # une des mesures effectivement utilisées est renseignée dans le périmètre.
    requested_school = question_requested_school(question)
    found_years = []

    for table in physical_tables:
        available_columns = set(schema_columns_from_sqlite(connection, table))

        usable_metrics = [
            metric
            for metric in metric_names
            if metric in available_columns
        ]

        if not usable_metrics:
            continue

        scope_clauses = []
        params = []

        if requested_school:
            scope_clauses.append("school_key = ?")
            params.append(requested_school)
        else:
            placeholders = ", ".join("?" for _ in POLYTECH_SCHOOLS)
            scope_clauses.append(f"school_key IN ({placeholders})")
            params.extend(POLYTECH_SCHOOLS)

        metric_clause = " OR ".join(
            f"{metric} IS NOT NULL"
            for metric in usable_metrics
        )

        where = " AND ".join(scope_clauses + [f"({metric_clause})"])

        try:
            row = connection.execute(
                f"SELECT MAX(year) FROM {table} WHERE {where}",
                params,
            ).fetchone()
        except sqlite3.Error:
            continue

        if row and row[0] is not None:
            found_years.append(int(row[0]))

    return max(found_years) if found_years else None


def schema_columns_from_sqlite(connection, table):
    """
    Colonnes réellement présentes dans SQLite.
    """
    try:
        return [
            row[1].lower()
            for row in connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        ]
    except sqlite3.Error:
        return []


def _component_is_populated(
    connection,
    table,
    metric,
    year,
    question,
    response_count=None,
    require_response=False,
):
    columns = set(schema_columns_from_sqlite(connection, table))

    if metric not in columns:
        return False

    if require_response and (
        not response_count
        or response_count not in columns
    ):
        return False

    where_scope, params = _validation_scope(
        connection,
        question,
        year,
    )

    conditions = [
        where_scope,
        f"{metric} IS NOT NULL",
    ]

    if require_response:
        conditions.extend([
            f"{response_count} IS NOT NULL",
            f"{response_count} > 0",
        ])

    try:
        count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {table}
            WHERE {" AND ".join(conditions)}
            """,
            params,
        ).fetchone()[0]
    except sqlite3.Error:
        return False

    return bool(count)


def validate_parallel_count_components(
    connection,
    question,
    sql,
    physical_tables,
    schemas,
    result_columns=None,
    result_rows=None,
):
    """
    Garde-fou générique pour les mesures additives de type count.

    Il détecte depuis les libellés du semantic_schema :
      - les populations parallèles d'une même mesure ;
      - leur éventuel response_count ;
      - les sous-groupes (Total, F, H, etc.).

    Il empêche :
      1) de choisir arbitrairement une seule population renseignée dans une
         question générale ;
      2) de mélanger « Total » avec un autre sous-groupe entre membres ;
      3) dans un calcul utilisant des response_count, d'utiliser une mesure
         sans son response_count associé.

    Aucune métrique métier particulière n'est connue par cette fonction.
    """
    specs = _semantic_column_specs(
        physical_tables,
        schemas,
    )

    count_specs = {
        name: spec
        for name, spec in specs.items()
        if (
            spec["type"] == "count"
            and spec["aggregation"] == "sum"
        )
    }

    referenced = {
        name
        for name in count_specs
        if _sql_references_name(sql, name)
    }

    if not referenced:
        return True, None

    # ------------------------------------------------------------------
    # Cohérence des sous-groupes : "Total" ne doit pas être mélangé avec
    # F/H/autre pour une même population + mesure.
    # ------------------------------------------------------------------
    groups = {}

    for name in referenced:
        spec = count_specs[name]
        key = (
            normalize_text(spec["population"]),
            normalize_text(spec["measure"]),
        )
        groups.setdefault(key, []).append(spec)

    for group_specs in groups.values():
        subgroups = {
            normalize_text(spec["subgroup"])
            for spec in group_specs
            if spec["subgroup"]
        }

        if "total" in subgroups and len(subgroups) > 1:
            labels = sorted({
                spec["subgroup"]
                for spec in group_specs
                if spec["subgroup"]
            })

            return False, (
                "La requête mélange plusieurs sous-groupes incompatibles "
                f"pour une même mesure : {', '.join(labels)}. "
                "Utilise le même sous-groupe dans tous les membres ; si une "
                "colonne Total manque dans un membre, n'utilise pas à sa place "
                "une colonne F/H/autre."
            )

    # ------------------------------------------------------------------
    # Déterminer si la requête est un ratio/taux : si des response_count
    # sémantiquement associés sont utilisés, ils doivent accompagner chaque
    # composante correspondante.
    # ------------------------------------------------------------------
    referenced_response_counts = {
        name
        for name, spec in specs.items()
        if (
            spec["type"] == "response_count"
            and _sql_references_name(sql, name)
        )
    }

    require_response = bool(referenced_response_counts)

    for name in referenced:
        spec = count_specs[name]
        weight = spec.get("response_count")

        if (
            require_response
            and weight
            and not _sql_references_name(sql, weight)
        ):
            return False, (
                f"La mesure '{name}' est utilisée dans un calcul qui mobilise "
                "des response_count, mais son response_count correspondant "
                f"'{weight}' est absent du SQL."
            )

    year = _infer_validation_year(
        connection,
        question,
        physical_tables,
        referenced,
        result_columns=result_columns,
        result_rows=result_rows,
    )

    if year is None:
        return True, None

    # Une population explicitement citée dans la question limite les pairs.
    mentioned_populations = {
        normalize_text(spec["population"])
        for spec in count_specs.values()
        if (
            spec["population"]
            and _question_mentions_population(
                question,
                spec["population"],
            )
        )
    }

    checked_groups = set()

    for name in referenced:
        spec = count_specs[name]

        key = (
            normalize_text(spec["measure"]),
            normalize_text(spec["subgroup"]),
        )

        if key in checked_groups:
            continue
        checked_groups.add(key)

        peers = [
            peer
            for peer in count_specs.values()
            if (
                normalize_text(peer["measure"]) == key[0]
                and normalize_text(peer["subgroup"]) == key[1]
                and peer["population"]
            )
        ]

        # La règle de couverture n'a de sens que s'il existe réellement
        # plusieurs populations parallèles dans le schéma.
        populations = {
            normalize_text(peer["population"])
            for peer in peers
        }

        if len(populations) <= 1:
            continue

        if mentioned_populations:
            peers = [
                peer
                for peer in peers
                if normalize_text(peer["population"])
                in mentioned_populations
            ]

        populated = []

        for peer in peers:
            usable = any(
                _component_is_populated(
                    connection,
                    table,
                    peer["name"],
                    year,
                    question,
                    response_count=peer.get("response_count"),
                    require_response=require_response,
                )
                for table in physical_tables
            )

            if usable:
                populated.append(peer)

        missing = [
            peer
            for peer in populated
            if not _sql_references_name(
                sql,
                peer["name"],
            )
        ]

        if missing:
            details = []

            for peer in missing:
                detail = peer["name"]

                if require_response and peer.get("response_count"):
                    detail += (
                        f" + {peer['response_count']}"
                    )

                details.append(detail)

            return False, (
                "La requête ignore des composantes parallèles de la même "
                f"mesure réellement renseignées en {year}: "
                + ", ".join(details)
                + ". La question ne limitant pas cette population, combine "
                  "toutes les composantes pertinentes au lieu d'en choisir "
                  "une seule."
            )

    return True, None


def validate_duplicate_result_rows(columns, rows):
    """
    Des lignes strictement identiques n'apportent aucune information
    supplémentaire et signalent souvent une jointure multipliée.
    """
    if not rows or len(rows) < 2:
        return True, None

    normalized = [
        tuple(row)
        for row in rows
    ]

    if len(set(normalized)) != len(normalized):
        return False, (
            "Le résultat contient des lignes strictement identiques. "
            "Supprime les doublons accidentels avec une jointure plus précise, "
            "DISTINCT ou une agrégation adaptée, sans supprimer de catégories "
            "réellement différentes."
        )

    return True, None


def _weighted_column_specs(physical_tables, schemas):
    """
    Construit, uniquement depuis le semantic_schema, les couples :
      mesure pondérable <-> response_count associé.

    Le rapprochement se fait par le chemin du libellé :
      population > mesure > sous-groupe
      population > Nombre de réponses > sous-groupe

    Aucune connaissance métier (salaire, FISE, etc.) n'est codée ici.
    """
    columns = {}

    for table in physical_tables:
        for column in parse_business_columns(schemas.get(table, "")):
            columns.setdefault(
                column["name"].lower(),
                column,
            )

    response_columns = [
        column
        for column in columns.values()
        if column.get("type") == "response_count"
    ]

    specs = {}

    for column in columns.values():
        if column.get("aggregation") != "weighted_average_by_response_count":
            continue

        parts = _label_parts(column.get("label", ""))

        if len(parts) < 3:
            continue

        population = " > ".join(parts[:-2])
        measure = parts[-2]
        subgroup = parts[-1]

        weight = None

        for candidate in response_columns:
            candidate_parts = _label_parts(candidate.get("label", ""))

            if len(candidate_parts) < 3:
                continue

            candidate_population = " > ".join(candidate_parts[:-2])
            candidate_subgroup = candidate_parts[-1]

            if (
                normalize_text(candidate_population)
                == normalize_text(population)
                and normalize_text(candidate_subgroup)
                == normalize_text(subgroup)
            ):
                weight = candidate["name"].lower()
                break

        specs[column["name"].lower()] = {
            "metric": column["name"].lower(),
            "weight": weight,
            "population": population,
            "measure": measure,
            "subgroup": subgroup,
        }

    return specs


def validate_weighted_component_coverage(
    connection,
    question,
    sql,
    physical_tables,
    schemas,
    result_columns=None,
    result_rows=None,
):
    """
    Vérifie qu'une synthèse pondérée générale ne choisit pas arbitrairement
    une seule composante parallèle lorsque plusieurs couples mesure/poids
    sont réellement renseignés dans le périmètre.

    Cette validation est entièrement pilotée par le semantic_schema :
    aggregation, type, libellés et disponibilité réelle des colonnes.
    """
    specs = _weighted_column_specs(
        physical_tables,
        schemas,
    )

    referenced_metrics = {
        name
        for name in specs
        if re.search(
            rf"\b{re.escape(name)}\b",
            sql,
            flags=re.IGNORECASE,
        )
    }

    if not referenced_metrics:
        return True, None

    # Chaque mesure pondérée doit utiliser le response_count correspondant.
    for metric in referenced_metrics:
        weight = specs[metric].get("weight")

        if weight and not re.search(
            rf"\b{re.escape(weight)}\b",
            sql,
            flags=re.IGNORECASE,
        ):
            return False, (
                f"La mesure pondérée '{metric}' doit utiliser son "
                f"response_count correspondant '{weight}'."
            )

    # Année réellement utilisée.
    years = explicit_years_in_question(question)

    if not years and result_columns and result_rows:
        try:
            year_index = [
                column.lower()
                for column in result_columns
            ].index("year")

            years = sorted({
                int(row[year_index])
                for row in result_rows
                if row[year_index] is not None
            })
        except (ValueError, TypeError):
            years = []

    # Sans année identifiable, pas de rejet basé sur une autre campagne.
    if len(years) != 1:
        return True, None

    year = years[0]
    requested_school = question_requested_school(question)

    clauses = ["year = ?"]
    base_params = [year]

    if requested_school:
        clauses.append("school_key = ?")
        base_params.append(requested_school)
    else:
        placeholders = ", ".join(
            "?"
            for _ in POLYTECH_SCHOOLS
        )
        clauses.append(
            f"school_key IN ({placeholders})"
        )
        base_params.extend(POLYTECH_SCHOOLS)

    where_sql = " AND ".join(clauses)

    # Si la question cite explicitement une population décrite par le schéma,
    # elle limite naturellement les composantes attendues.
    normalized_question = normalize_text(question)

    mentioned_populations = {
        normalize_text(spec["population"])
        for spec in specs.values()
        if spec["population"]
        and normalize_text(spec["population"]) in normalized_question
    }

    # Même mesure sémantique + même sous-groupe = composantes parallèles.
    for metric in referenced_metrics:
        spec = specs[metric]

        peers = [
            peer
            for peer in specs.values()
            if (
                normalize_text(peer["measure"])
                == normalize_text(spec["measure"])
                and normalize_text(peer["subgroup"])
                == normalize_text(spec["subgroup"])
            )
        ]

        if mentioned_populations:
            peers = [
                peer
                for peer in peers
                if normalize_text(peer["population"])
                in mentioned_populations
            ]

        populated_peers = []

        for peer in peers:
            if not peer.get("weight"):
                continue

            usable = 0

            for table in physical_tables:
                try:
                    usable += int(
                        connection.execute(
                            f"""
                            SELECT COUNT(*)
                            FROM {table}
                            WHERE {where_sql}
                              AND {peer['metric']} IS NOT NULL
                              AND {peer['weight']} IS NOT NULL
                              AND {peer['weight']} > 0
                            """,
                            base_params,
                        ).fetchone()[0]
                        or 0
                    )
                except sqlite3.Error:
                    continue

            if usable > 0:
                populated_peers.append(peer)

        missing = [
            peer
            for peer in populated_peers
            if not re.search(
                rf"\b{re.escape(peer['metric'])}\b",
                sql,
                flags=re.IGNORECASE,
            )
        ]

        if missing:
            details = ", ".join(
                f"{peer['metric']} pondéré par {peer['weight']}"
                for peer in missing
            )

            return False, (
                "La requête calcule une synthèse "
                "weighted_average_by_response_count mais ignore des "
                "composantes parallèles réellement renseignées pour le même "
                f"indicateur et le même sous-groupe en {year}: {details}. "
                "La question ne limitant pas ce périmètre, combine toutes "
                "les composantes pertinentes dans un même numérateur "
                "SUM(valeur * poids) et un même dénominateur SUM(poids)."
            )

    return True, None


def build_question_data_profile(
    connection,
    question,
    physical_tables,
    schemas,
):
    """
    Profil compact des colonnes métier réellement renseignées dans le
    périmètre demandé (école/réseau + année explicite éventuelle).

    Aucune formule métier n'est décidée ici.
    """
    requested_school = question_requested_school(question)
    years = explicit_years_in_question(question)

    clauses = []
    params = []

    if requested_school:
        clauses.append("school_key = ?")
        params.append(requested_school)
    else:
        placeholders = ", ".join("?" for _ in POLYTECH_SCHOOLS)
        clauses.append(f"school_key IN ({placeholders})")
        params.extend(POLYTECH_SCHOOLS)

    if len(years) == 1:
        clauses.append("year = ?")
        params.append(years[0])

    where_sql = " WHERE " + " AND ".join(clauses)

    stats = {}

    for table in physical_tables:
        schema = schemas.get(table, "")
        for column in parse_business_columns(schema):
            name = column.get("name", "")

            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                continue

            item = stats.setdefault(
                name,
                {
                    "label": column.get("label", ""),
                    "type": column.get("type", ""),
                    "aggregation": column.get("aggregation", ""),
                    "non_null": 0,
                },
            )

            try:
                count = connection.execute(
                    f"SELECT COUNT({name}) FROM {table}{where_sql}",
                    params,
                ).fetchone()[0]
            except sqlite3.Error:
                continue

            item["non_null"] += int(count or 0)

    populated = [
        (name, info)
        for name, info in stats.items()
        if info["non_null"] > 0
    ]

    if not populated:
        return (
            "Aucune colonne métier non NULL détectée dans le périmètre "
            "demandé."
        )

    lines = []
    if requested_school:
        lines.append(f"- périmètre école: {requested_school}")
    else:
        lines.append("- périmètre: 16 écoles du réseau Polytech")

    if len(years) == 1:
        lines.append(f"- année: {years[0]}")

    for name, info in populated:
        metadata = []
        if info["label"]:
            metadata.append(f"libellé={info['label']}")
        if info["type"]:
            metadata.append(f"type={info['type']}")
        if info["aggregation"]:
            metadata.append(f"agrégation={info['aggregation']}")

        suffix = " | ".join(metadata)
        if suffix:
            suffix = " | " + suffix

        lines.append(
            f"- {name}: non_NULL={info['non_null']}{suffix}"
        )

    return "\n".join(lines)


def question_requested_school(question):
    """
    Retourne le school_key d'une école du réseau lorsqu'elle est explicitement
    citée dans la question. Sinon retourne None.

    Ceci encode uniquement le périmètre applicatif du réseau, pas une règle métier.
    """
    normalized = normalize_text(question)
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    aliases = {
        school: normalize_text(school).replace("_", " ")
        for school in POLYTECH_SCHOOLS
    }

    for school, alias in aliases.items():
        if re.search(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
            normalized,
        ):
            return school

    return None


def _cte_school_values(sql):
    """
    Extrait les valeurs d'un CTE du type :
        ecoles(school_key) AS (VALUES ('A'), ('B'), ...)
    afin que le validateur de périmètre reconnaisse aussi cette écriture SQL.
    """
    values = set()

    pattern = re.compile(
        r"""
        \b[a-zA-Z_][a-zA-Z0-9_]*\s*
        \(\s*school_key\s*\)\s*
        AS\s*\(\s*VALUES\s*
        (?P<body>.*?)
        \)\s*
        (?=,\s*[a-zA-Z_][a-zA-Z0-9_]*\s+AS|\bSELECT\b)
        """,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

    for match in pattern.finditer(sql):
        values.update(
            value.strip().upper()
            for value in re.findall(
                r"'([^']+)'",
                match.group("body"),
            )
        )

    return values


def validate_implicit_year_scope(question, sql, physical_tables, schemas):
    """
    Sans année explicite, une requête historisée doit sélectionner une seule
    campagne récente avant l'agrégation.
    """
    if explicit_years_in_question(question):
        return True, None

    has_year_dimension = any(
        re.search(
            r"(?m)^\s*-\s+year\s+INTEGER\b",
            schemas.get(table, ""),
        )
        for table in physical_tables
    )

    if not has_year_dimension:
        return True, None

    max_year = re.search(
        r"\bMAX\s*\(\s*(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?year\s*\)",
        sql,
        flags=re.IGNORECASE,
    )

    latest_order = re.search(
        r"\bORDER\s+BY\s+(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?year\s+DESC\b"
        r"[\s\S]{0,300}?\bLIMIT\s+1\b",
        sql,
        flags=re.IGNORECASE,
    )

    if not max_year and not latest_order:
        return False, (
            "Aucune année n'est explicitement demandée, mais les tables sont "
            "historisées. Sélectionne une seule campagne récente exploitable "
            "avant l'agrégation (MAX(year) ou ORDER BY year DESC LIMIT 1) ; "
            "ne mélange pas plusieurs campagnes."
        )

    return True, None


def validate_alias_semantic_shadowing(sql, physical_tables, schemas):
    """
    Empêche une dimension générique d'être renommée avec le nom d'une vraie
    colonne métier dédiée.
    """
    business_names = {
        column["name"].lower()
        for table in physical_tables
        for column in parse_business_columns(
            schemas.get(table, "")
        )
    }

    aliases = re.findall(
        r"\b(row_name|row_group|school_key|year)\b\s+AS\s+"
        r"([a-zA-Z_][a-zA-Z0-9_]*)",
        sql,
        flags=re.IGNORECASE,
    )

    for dimension, alias in aliases:
        alias_lower = alias.lower()

        if alias_lower in business_names and alias_lower != dimension.lower():
            return False, (
                f"La requête renomme '{dimension}' en '{alias}', alors qu'une "
                f"colonne métier dédiée '{alias_lower}' existe. Utilise cette "
                "colonne métier au lieu de changer le sens de la dimension."
            )

    return True, None


def _question_requests_virtual_member_identity(question):
    tokens = set(text_tokens(question))

    return bool(
        {
            "specialite",
            "specialites",
            "filiere",
            "filieres",
        }
        & tokens
    )


def _member_id_from_table(table):
    match = re.search(
        r"_d(\d+)(?:_|$)",
        table,
        flags=re.IGNORECASE,
    )
    return f"d{match.group(1)}" if match else None


def validate_virtual_member_identity(
    question,
    selected_entries,
    columns,
    rows,
    sql="",
):
    """
    Pour une demande de spécialité/filière sur une famille D* :
      - le résultat doit conserver un member_id Dn ;
      - sans école explicite, il doit aussi conserver school_key ;
      - le GROUP BY doit garder school_key afin de ne jamais fusionner
        le même numéro D entre plusieurs écoles.
    """
    if not _question_requests_virtual_member_identity(question):
        return True, None

    virtual_members = {
        _member_id_from_table(table)
        for entry in selected_entries
        if "*" in entry.get("table", "")
        for table in entry.get("members", [])
    }
    virtual_members.discard(None)

    if not virtual_members or not rows:
        return True, None

    lowered_columns = [
        normalize_text(column)
        for column in (columns or [])
    ]

    member_column_indexes = []

    for index in range(len(columns or [])):
        values = {
            normalize_text(row[index])
            for row in rows
            if (
                index < len(row)
                and isinstance(row[index], str)
            )
        }

        if values & virtual_members:
            member_column_indexes.append(index)

    if not member_column_indexes:
        return False, (
            "La question demande d'identifier/comparer une spécialité ou "
            "filière d'une famille D*, mais le résultat ne conserve aucun "
            "member_id Dn. Ajoute une constante comme 'd3' AS member_id dans "
            "chaque branche physique."
        )

    requested_school = question_requested_school(question)

    if requested_school:
        return True, None

    if "school_key" not in lowered_columns:
        return False, (
            "Aucune école précise n'est demandée. Un member_id Dn n'est pas "
            "une spécialité globale : l'identité doit être le couple "
            "(school_key, member_id). Retourne school_key avec member_id et "
            "classe/groupe par ce couple."
        )

    # Vérifier que le SQL ne groupe pas seulement par member_id.
    group_by_clauses = re.findall(
        r"\bGROUP\s+BY\s+(.+?)(?=\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|\)|;|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if group_by_clauses:
        has_school_group = any(
            re.search(
                r"\bschool_key\b",
                clause,
                flags=re.IGNORECASE,
            )
            for clause in group_by_clauses
        )

        if not has_school_group:
            return False, (
                "La requête compare des membres D* sur plusieurs écoles mais "
                "GROUP BY ne conserve pas school_key. Les d1/d2/d3... de deux "
                "écoles différentes ne représentent pas la même spécialité. "
                "Groupe par school_key + member_id, puis classe ces couples."
            )

    return True, None


def _weighted_metrics_used(sql, physical_tables, schemas):
    metrics = []

    for table in physical_tables:
        for column in parse_business_columns(
            schemas.get(table, "")
        ):
            if (
                column.get("aggregation")
                == "weighted_average_by_response_count"
                and _sql_references_name(
                    sql,
                    column["name"].lower(),
                )
            ):
                metrics.append(
                    column["name"].lower()
                )

    return sorted(set(metrics))


def validate_weighted_dimension_scope(
    question,
    sql,
    physical_tables,
    schemas,
    columns,
    rows,
):
    """
    Une mesure pondérée ne doit pas fusionner silencieusement plusieurs
    row_name distincts.
    """
    if not _weighted_metrics_used(
        sql,
        physical_tables,
        schemas,
    ):
        return True, None

    observed = allowed_dimension_values(
        physical_tables,
        schemas,
        "row_name",
    )

    if len(observed) <= 1:
        return True, None

    normalized_question = normalize_text(question)

    mentioned = [
        value
        for value in observed
        if (
            normalize_text(value)
            and normalize_text(value)
            in normalized_question
        )
    ]

    equality_values = re.findall(
        r"\brow_name\b\s*=\s*'([^']*)'",
        sql,
        flags=re.IGNORECASE,
    )

    in_blocks = re.findall(
        r"\brow_name\b\s+IN\s*\(([^)]*)\)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    filtered_values = list(equality_values)

    for block in in_blocks:
        filtered_values.extend(
            re.findall(r"'([^']*)'", block)
        )

    filtered_normalized = {
        normalize_text(value)
        for value in filtered_values
    }

    if mentioned:
        missing = [
            value
            for value in mentioned
            if normalize_text(value)
            not in filtered_normalized
        ]

        if missing:
            return False, (
                "La question sélectionne explicitement la catégorie row_name "
                f"{', '.join(missing)}. Filtre-la explicitement dans le SQL au "
                "lieu de calculer aussi les autres catégories."
            )

        return True, None

    observed_normalized = {
        normalize_text(value)
        for value in observed
    }

    for index, _column in enumerate(columns or []):
        values = {
            normalize_text(row[index])
            for row in (rows or [])
            if (
                index < len(row)
                and isinstance(row[index], str)
                and normalize_text(row[index])
            )
        }

        if values and values.issubset(
            observed_normalized
        ):
            return True, None

    return False, (
        "Plusieurs row_name distincts existent pour cette mesure pondérée et "
        "la question n'en choisit aucun. Conserve row_name dans le résultat et "
        "calcule une valeur séparée par catégorie."
    )


def validate_null_safe_parallel_addition(
    sql,
    physical_tables,
    schemas,
):
    """
    Les additions a+b entre counts parallèles doivent être NULL-safe.
    """
    specs = _semantic_column_specs(
        physical_tables,
        schemas,
    )

    count_specs = [
        spec
        for spec in specs.values()
        if (
            spec.get("type") == "count"
            and spec.get("aggregation") == "sum"
        )
    ]

    groups = {}

    for spec in count_specs:
        key = (
            normalize_text(
                spec.get("measure", "")
            ),
            normalize_text(
                spec.get("subgroup", "")
            ),
        )
        groups.setdefault(key, []).append(spec)

    for peers in groups.values():
        populations = {
            normalize_text(
                peer.get("population", "")
            )
            for peer in peers
            if peer.get("population")
        }

        if len(populations) <= 1:
            continue

        names = [
            peer["name"]
            for peer in peers
        ]

        for i, left in enumerate(names):
            for right in names[i + 1:]:
                direct = re.search(
                    rf"\b{re.escape(left)}\b\s*\+\s*\b{re.escape(right)}\b"
                    rf"|\b{re.escape(right)}\b\s*\+\s*\b{re.escape(left)}\b",
                    sql,
                    flags=re.IGNORECASE,
                )

                if not direct:
                    continue

                safe_left = re.search(
                    rf"COALESCE\s*\(\s*{re.escape(left)}\s*,\s*0(?:\.0+)?\s*\)",
                    sql,
                    flags=re.IGNORECASE,
                )
                safe_right = re.search(
                    rf"COALESCE\s*\(\s*{re.escape(right)}\s*,\s*0(?:\.0+)?\s*\)",
                    sql,
                    flags=re.IGNORECASE,
                )

                if not (safe_left and safe_right):
                    return False, (
                        f"La requête additionne directement '{left}' et "
                        f"'{right}'. Utilise COALESCE({left}, 0) + "
                        f"COALESCE({right}, 0) pour ne pas perdre les lignes "
                        "où une seule composante est renseignée."
                    )

    return True, None


def build_virtual_member_label_context(
    connection,
    question,
    selected_entries,
    result_columns,
    result_rows,
):
    """
    Résout les libellés humains des couples (school_key, member_id) réellement
    retournés. Un numéro D n'est jamais interprété indépendamment de l'école.
    """
    if not _question_requests_virtual_member_identity(question):
        return ""

    requested_school = question_requested_school(question)

    lowered_columns = [
        normalize_text(column)
        for column in (result_columns or [])
    ]

    school_index = (
        lowered_columns.index("school_key")
        if "school_key" in lowered_columns
        else None
    )

    member_indexes = []

    for index in range(len(result_columns or [])):
        values = {
            normalize_text(row[index])
            for row in (result_rows or [])
            if (
                index < len(row)
                and isinstance(row[index], str)
            )
        }

        if any(
            re.fullmatch(r"d\d+", value)
            for value in values
        ):
            member_indexes.append(index)

    if not member_indexes:
        return ""

    pairs = set()

    for row in result_rows or []:
        school = requested_school

        if school_index is not None and school_index < len(row):
            value = row[school_index]
            if isinstance(value, str):
                upper = value.strip().upper()
                if upper in POLYTECH_SCHOOLS:
                    school = upper

        if not school:
            continue

        for member_index in member_indexes:
            if member_index >= len(row):
                continue

            value = row[member_index]

            if not isinstance(value, str):
                continue

            member_id = normalize_text(value)

            if re.fullmatch(r"d\d+", member_id):
                pairs.add(
                    (school, member_id)
                )

    if not pairs:
        return ""

    lines = [
        "LIBELLÉS HUMAINS DES MEMBRES D* "
        "(identité = school_key + member_id):"
    ]

    for school, member_id in sorted(
        pairs,
        key=lambda item: (
            item[0],
            int(item[1][1:]),
        ),
    ):
        base_table = f"cti_viii_{member_id}"

        exists = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name=?",
            (base_table,),
        ).fetchone()

        if not exists:
            continue

        try:
            row = connection.execute(
                f"""
                SELECT section_title
                FROM {base_table}
                WHERE school_key = ?
                  AND section_title IS NOT NULL
                  AND TRIM(section_title) <> ''
                ORDER BY year DESC
                LIMIT 1
                """,
                (school,),
            ).fetchone()
        except sqlite3.Error:
            row = None

        if row and row[0]:
            lines.append(
                f"- {school} + {member_id}: {row[0]}"
            )
        else:
            lines.append(
                f"- {school} + {member_id}: libellé humain non résolu"
            )

    return "\n".join(lines) if len(lines) > 1 else ""




def validate_school_scope(question, sql):
    """
    Garantit le périmètre applicatif :
      - école explicitement demandée -> cette école doit être filtrée ;
      - aucune école demandée -> la requête doit être restreinte exactement
        aux school_key du réseau Polytech.

    Le validateur ne connaît aucun indicateur métier.
    """
    requested = question_requested_school(question)

    # Toutes les constantes school_key présentes dans les filtres SQL.
    equality_values = re.findall(
        r"\bschool_key\b\s*=\s*'([^']+)'",
        sql,
        flags=re.IGNORECASE,
    )

    in_blocks = re.findall(
        r"\bschool_key\b\s+IN\s*\(([^)]*)\)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    in_values = []
    for block in in_blocks:
        in_values.extend(
            re.findall(r"'([^']+)'", block)
        )

    literal_school_values = {
        value.strip().upper()
        for value in equality_values + in_values
    }

    literal_school_values.update(
        _cte_school_values(sql)
    )

    network = set(POLYTECH_SCHOOLS)

    if requested:
        if requested not in literal_school_values:
            return False, (
                f"La question porte explicitement sur {requested}, mais le SQL "
                f"ne restreint pas school_key à cette école."
            )

        outside = literal_school_values - {requested}
        if outside:
            return False, (
                f"La question porte sur {requested}, mais le SQL référence aussi "
                f"d'autres school_key : {', '.join(sorted(outside))}."
            )

        return True, None

    # Sans école explicite, il faut le réseau exact, pas toute la base.
    if literal_school_values != network:
        missing = sorted(network - literal_school_values)
        extra = sorted(literal_school_values - network)

        details = []
        if missing:
            details.append("manquantes: " + ", ".join(missing))
        if extra:
            details.append("hors réseau: " + ", ".join(extra))
        if not literal_school_values:
            details.append(
                "aucun filtre school_key explicite : la requête inclut "
                "potentiellement des écoles hors réseau"
            )

        return False, (
            "Aucune école précise n'est demandée : le périmètre doit être "
            "EXACTEMENT les 16 school_key du réseau Polytech. "
            + " ; ".join(details)
        )

    return True, None


def validate_virtual_family_coverage(
    connection,
    question,
    sql,
    selected_entries,
    schemas,
):
    """
    Exige seulement les membres D* qui peuvent réellement contribuer aux
    colonnes utilisées et qui contiennent des données dans le périmètre.
    """
    used_tables = {
        table.lower()
        for table in referenced_physical_tables(sql)
    }

    sql_tokens = {
        token.lower()
        for token in re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
            sql,
        )
    }

    for entry in selected_entries:
        family = entry.get("table", "")
        members = [
            table
            for table in entry.get("members", [])
            if table in schemas
        ]

        if "*" not in family or len(members) <= 1:
            continue

        used_family_members = [
            table
            for table in members
            if table.lower() in used_tables
        ]

        # Une famille peut avoir été sélectionnée uniquement pour fournir du
        # contexte sémantique / un libellé humain. Si le SQL ne la référence
        # pas du tout, elle ne doit pas déclencher une obligation de couverture
        # de tous ses membres physiques.
        if not used_family_members:
            continue

        missing = [
            table
            for table in members
            if table.lower() not in used_tables
        ]

        if not missing:
            continue

        family_business = {
            column["name"].lower()
            for table in members
            for column in parse_business_columns(
                schemas.get(table, "")
            )
        }

        referenced_business = (
            family_business & sql_tokens
        )

        if not referenced_business:
            return False, (
                f"La famille virtuelle {family} omet des membres physiques : "
                f"{', '.join(missing)}."
            )

        year = _infer_validation_year(
            connection,
            question,
            members,
            referenced_business,
        )

        required_missing = []

        for table in missing:
            sqlite_columns = set(
                schema_columns_from_sqlite(
                    connection,
                    table,
                )
            )

            candidate_columns = [
                column
                for column in referenced_business
                if column in sqlite_columns
            ]

            if not candidate_columns:
                continue

            if year is None:
                required_missing.append(table)
                continue

            where_scope, params = _validation_scope(
                connection,
                question,
                year,
            )

            populated = False

            for column in candidate_columns:
                try:
                    count = connection.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM {table}
                        WHERE {where_scope}
                          AND {column} IS NOT NULL
                        """,
                        params,
                    ).fetchone()[0]
                except sqlite3.Error:
                    count = 0

                if count:
                    populated = True
                    break

            if populated:
                required_missing.append(table)

        if required_missing:
            return False, (
                f"La famille virtuelle {family} omet des membres qui ont des "
                "données utiles pour les colonnes utilisées : "
                f"{', '.join(required_missing)}. Ajoute ces membres utiles. "
                "Les membres sans colonne compatible ou sans donnée exploitable "
                "n'ont pas à être ajoutés."
            )

    return True, None



def build_virtual_family_coverage_context(
    connection,
    question,
    sql,
    selected_entries,
    schemas,
):
    """
    Produit un contexte explicatif lorsqu'une famille D* utilisée par le SQL
    n'emploie pas tous ses membres physiques.

    Le contexte distingue notamment :
      - membre sans colonne compatible avec les colonnes réellement utilisées ;
      - membre avec colonnes compatibles mais sans donnée exploitable dans le
        périmètre/campagne.

    Ce contexte n'invente aucune valeur et ne décide aucune règle métier.
    """
    used_tables = {
        table.lower()
        for table in referenced_physical_tables(sql)
    }

    sql_tokens = {
        token.lower()
        for token in re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
            sql,
        )
    }

    notes = []

    for entry in selected_entries:
        family = entry.get("table", "")
        members = [
            table
            for table in entry.get("members", [])
            if table in schemas
        ]

        if "*" not in family or len(members) <= 1:
            continue

        used_family_members = [
            table
            for table in members
            if table.lower() in used_tables
        ]

        # Famille sélectionnée comme contexte uniquement.
        if not used_family_members:
            continue

        omitted = [
            table
            for table in members
            if table.lower() not in used_tables
        ]

        if not omitted:
            continue

        family_business = {
            column["name"].lower()
            for table in members
            for column in parse_business_columns(
                schemas.get(table, "")
            )
        }

        referenced_business = (
            family_business & sql_tokens
        )

        year = _infer_validation_year(
            connection,
            question,
            members,
            referenced_business,
        )

        for table in omitted:
            sqlite_columns = set(
                schema_columns_from_sqlite(
                    connection,
                    table,
                )
            )

            compatible_columns = sorted(
                referenced_business & sqlite_columns
            )

            if not compatible_columns:
                notes.append(
                    {
                        "family": family,
                        "table": table,
                        "reason": (
                            "aucune des colonnes métier utilisées par le calcul "
                            "n'existe dans ce membre avec le même nom/structure"
                        ),
                    }
                )
                continue

            if year is None:
                notes.append(
                    {
                        "family": family,
                        "table": table,
                        "reason": (
                            "membre non utilisé ; la campagne exploitable n'a "
                            "pas pu être déterminée pour vérifier ses données"
                        ),
                    }
                )
                continue

            where_scope, params = _validation_scope(
                connection,
                question,
                year,
            )

            populated_columns = []

            for column in compatible_columns:
                try:
                    count = connection.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM {table}
                        WHERE {where_scope}
                          AND {column} IS NOT NULL
                        """,
                        params,
                    ).fetchone()[0]
                except sqlite3.Error:
                    count = 0

                if count:
                    populated_columns.append(column)

            if populated_columns:
                # En principe ce cas aurait dû être bloqué par le validateur.
                reason = (
                    "des colonnes compatibles et renseignées existent mais "
                    "le membre n'apparaît pas dans le SQL"
                )
            else:
                reason = (
                    "les colonnes compatibles ne sont pas renseignées dans "
                    f"le périmètre/campagne {year}"
                )

            notes.append(
                {
                    "family": family,
                    "table": table,
                    "reason": reason,
                }
            )

    if not notes:
        return ""

    lines = [
        "COUVERTURE PARTIELLE DE FAMILLES D* :",
        (
            "Le calcul SQL n'utilise pas tous les membres physiques de la/les "
            "famille(s) ci-dessous. La valeur ne doit donc pas être présentée "
            "comme exhaustive sur tous les membres si des exclusions "
            "structurelles existent."
        ),
    ]

    for note in notes:
        lines.append(
            f"- {note['family']} / {note['table']}: {note['reason']}."
        )

    return "\n".join(lines)



def validate_dimension_filters(sql, physical_tables, schemas):
    """
    Garde-fou déterministe : une valeur littérale utilisée pour row_name
    ou row_group doit réellement exister dans le semantic_schema.
    """
    for dimension in ("row_name", "row_group"):
        observed = allowed_dimension_values(
            physical_tables,
            schemas,
            dimension,
        )

        observed_normalized = {
            normalize_text(value): value
            for value in observed
        }

        equal_values = re.findall(
            rf"(?:\b[a-zA-Z_][a-zA-Z0-9_]*\.)?"
            rf"\b{dimension}\b\s*=\s*'((?:''|[^'])*)'",
            sql,
            flags=re.IGNORECASE,
        )

        in_blocks = re.findall(
            rf"(?:\b[a-zA-Z_][a-zA-Z0-9_]*\.)?"
            rf"\b{dimension}\b\s+IN\s*\(((?:'(?:''|[^'])*'|[^')])*)\)",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )

        in_values = []
        for block in in_blocks:
            in_values.extend(
                re.findall(r"'((?:''|[^'])*)'", block)
            )

        for literal in equal_values + in_values:
            value = literal.replace("''", "'")
            if normalize_text(value) not in observed_normalized:
                if not observed:
                    return False, (
                        f"Le SQL filtre {dimension}='{value}', mais aucune "
                        f"valeur observée de {dimension} n'est fournie dans "
                        f"le schéma sélectionné."
                    )

                sample = " | ".join(observed[:12])
                return False, (
                    f"Valeur non observée pour {dimension}: {value}. "
                    f"Valeurs disponibles, par exemple : {sample}"
                )

    return True, None


def validate_sql(
    connection,
    sql,
    physical_tables,
    schemas,
):
    stripped = sql.strip()

    if not re.match(
        r"^(SELECT|WITH)\b",
        stripped,
        flags=re.IGNORECASE,
    ):
        return False, (
            "Seules les requêtes SELECT / WITH sont autorisées."
        )

    forbidden = re.search(
        r"\b("
        r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
        r"ATTACH|DETACH|VACUUM|PRAGMA"
        r")\b",
        stripped,
        flags=re.IGNORECASE,
    )

    if forbidden:
        return False, (
            f"Instruction SQL interdite : {forbidden.group(1)}"
        )

    allowed = {
        table.lower()
        for table in physical_tables
    }

    used = {
        table.lower()
        for table in referenced_physical_tables(sql)
    }

    unauthorized = sorted(used - allowed)

    if unauthorized:
        return False, (
            "Table hors du schéma sélectionné : "
            + ", ".join(unauthorized)
        )

    valid, error = validate_dimension_filters(
        sql,
        physical_tables,
        schemas,
    )

    if not valid:
        return False, error

    try:
        connection.execute(
            "EXPLAIN QUERY PLAN " + stripped
        )
    except sqlite3.Error as exc:
        return False, str(exc)

    return True, None


def result_is_usable(rows, columns):
    if not rows:
        return False

    technical = {
        "school_key",
        "year",
        "row_name",
        "row_group",
    }

    useful_indexes = [
        index
        for index, column in enumerate(columns or [])
        if column.lower() not in technical
    ]

    if useful_indexes:
        return any(
            row[index] is not None
            for row in rows
            for index in useful_indexes
        )

    return any(
        value is not None
        for row in rows
        for value in row
    )


def extract_school_scope_from_sql(sql):
    """
    Récupère les school_key explicitement utilisés dans le SQL.
    Ce n'est qu'un indice de périmètre pour profiler les données.
    """
    values = re.findall(
        r"\bschool_key\b\s*=\s*'([^']+)'",
        sql,
        flags=re.IGNORECASE,
    )

    result = []
    for value in values:
        value = value.strip()
        if value and value not in result:
            result.append(value)

    return result


def build_runtime_data_profile(
    connection,
    physical_tables,
    schemas,
    sql,
    max_samples=3,
):
    """
    Produit un petit profil des DONNÉES RÉELLEMENT PRÉSENTES dans SQLite.

    Il sert seulement lorsqu'un SQL valide retourne un résultat inutilisable.
    GPT peut alors distinguer :
      - une colonne présente dans le schéma mais vide dans le périmètre ;
      - une colonne réellement renseignée ;
      - des artefacts texte éventuels dans une colonne censée être numérique.

    Aucune décision métier n'est codée ici.
    """
    schools = extract_school_scope_from_sql(sql)

    where_sql = ""
    params = []

    if schools:
        placeholders = ", ".join("?" for _ in schools)
        where_sql = f" WHERE school_key IN ({placeholders})"
        params = schools

    family_stats = {}
    years = set()
    total_rows = 0

    for table in physical_tables:
        schema = schemas.get(table, "")
        columns = schema_columns(schema)

        if not columns:
            continue

        try:
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {table}{where_sql}",
                params,
            ).fetchone()[0]
        except sqlite3.Error:
            continue

        if not row_count:
            continue

        total_rows += int(row_count)

        try:
            table_years = connection.execute(
                f"""
                SELECT DISTINCT year
                FROM {table}
                {where_sql}
                AND year IS NOT NULL
                """ if where_sql else
                f"""
                SELECT DISTINCT year
                FROM {table}
                WHERE year IS NOT NULL
                """,
                params,
            ).fetchall()

            for (year,) in table_years:
                if year is not None:
                    years.add(year)
        except sqlite3.Error:
            pass

        for column in columns:
            stats = family_stats.setdefault(
                column,
                {
                    "non_null": 0,
                    "numeric": 0,
                    "text": 0,
                    "samples": [],
                },
            )

            try:
                non_null, numeric_count, text_count = connection.execute(
                    f"""
                    SELECT
                        SUM(CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END),
                        SUM(CASE WHEN typeof({column}) IN ('integer','real')
                                 AND {column} IS NOT NULL THEN 1 ELSE 0 END),
                        SUM(CASE WHEN typeof({column}) = 'text'
                                 AND {column} IS NOT NULL THEN 1 ELSE 0 END)
                    FROM {table}
                    {where_sql}
                    """,
                    params,
                ).fetchone()

                stats["non_null"] += int(non_null or 0)
                stats["numeric"] += int(numeric_count or 0)
                stats["text"] += int(text_count or 0)

                if len(stats["samples"]) < max_samples:
                    sample_rows = connection.execute(
                        f"""
                        SELECT DISTINCT CAST({column} AS TEXT)
                        FROM {table}
                        {where_sql + (' AND ' if where_sql else ' WHERE ')}
                        {column} IS NOT NULL
                        LIMIT ?
                        """,
                        [*params, max_samples],
                    ).fetchall()

                    for (sample,) in sample_rows:
                        if (
                            sample is not None
                            and sample not in stats["samples"]
                            and len(stats["samples"]) < max_samples
                        ):
                            stats["samples"].append(sample)

            except sqlite3.Error:
                continue

    lines = [
        "PROFIL DES DONNÉES RÉELLEMENT PRÉSENTES DANS LE PÉRIMÈTRE DU SQL:",
        f"- lignes physiques inspectées: {total_rows}",
    ]

    if schools:
        lines.append("- school_key: " + ", ".join(schools))

    if years:
        lines.append(
            "- années présentes: "
            + ", ".join(str(year) for year in sorted(years))
        )

    for column, stats in family_stats.items():
        if stats["non_null"] == 0:
            availability = "aucune valeur non NULL"
        else:
            availability = (
                f"non_NULL={stats['non_null']}, "
                f"numériques={stats['numeric']}, "
                f"textes={stats['text']}"
            )

        sample_text = ""
        if stats["samples"]:
            sample_text = " | exemples=" + ", ".join(
                repr(sample)
                for sample in stats["samples"]
            )

        lines.append(
            f"- {column}: {availability}{sample_text}"
        )

    return "\n".join(lines)


def failed_result_context(columns, rows):
    """
    Petit aperçu du résultat SQL précédent envoyé au correcteur.
    """
    if columns is None or rows is None:
        return "Aucun résultat SQLite disponible."

    return (
        "COLONNES DU RÉSULTAT PRÉCÉDENT:\n"
        + json.dumps(columns, ensure_ascii=False)
        + "\nRÉSULTAT PRÉCÉDENT (max 5 lignes):\n"
        + json.dumps(rows[:5], ensure_ascii=False, default=str)
    )


# ---------------------------------------------------------------------------
# Correction SQL par GPT
# ---------------------------------------------------------------------------

def correct_sql(
    llm,
    question,
    sql,
    error,
    selected_entries,
    schemas,
    runtime_profile="",
    previous_result="",
):
    schema_context = build_sql_context(
        selected_entries,
        schemas,
    )

    schools = ", ".join(POLYTECH_SCHOOLS)

    prompt = f"""
QUESTION:
{question}

SQL À CORRIGER:
{sql}

PROBLÈME DÉTECTÉ:
{error}

{previous_result}

{runtime_profile}

SCHÉMA SÉMANTIQUE AUTORISÉ:
{schema_context}

SCHOOL_KEY DU RÉSEAU POLYTECH:
{schools}

Régénère entièrement une requête SQLite correcte.

Contraintes:
- SELECT/WITH uniquement ;
- mêmes familles sélectionnées : ne change pas de sujet ;
- pour une synthèse générale d'apprenants/diplômés sans restriction de voie,
  vérifie chaque couple membre physique × voie : inclure FISE et FISA + FISEA
  dans tous les membres où elles sont compatibles et disponibles. Si FISE n'est
  lue que dans D1 et l'autre voie dans tous les Dn, ajoute les branches FISE
  manquantes, sans changer les chiffres sources. Pour un taux ou une pondération,
  conserve source_table = vrai nom physique et voie = libellé de la voie dans
  chaque branche ; déduplique et valide les couples mesure/réponses ou valeur/poids
  par voie avant la somme globale. N'écarte pas une voie exploitable parce que
  l'autre est absente, et n'assimile pas une valeur absente à un zéro observé.
  Une couverture de toutes les écoles ne dispense pas de vérifier toutes les
  voies et tous les membres. Garde le signalement des incohérences sources ;
- sauf ventilation ou école explicitement demandée, vise un résultat global du
  réseau et non une ligne par école. Un ratio global se calcule après sommation
  des effectifs compatibles ; une proportion globale inclut le facteur 100.
  Si les familles sélectionnées n'ont pas les ingrédients, explique cette limite
  sans présenter une ventilation par école comme une synthèse globale ;
- un horizon absent de la question ne doit pas être inventé ; conserve le
  contexte réel de la mesure sélectionnée et nomme-le précisément. Un horizon
  explicitement demandé ne peut pas être remplacé pour augmenter la couverture ;
- ne somme jamais des colonnes percentage/percent entre écoles ni ne transforme
  ces sommes en effectifs. Un ratio global de catégories nécessite les effectifs
  de chaque catégorie, ou des pourcentages pondérés par les dénominateurs exacts
  de même population/période. Sans effectifs ou poids compatibles disponibles,
  retourne les parts par école et expose l'impossibilité du calcul global.
  Un quotient de sommes de pourcentages non pondérés ne remplace pas un ratio
  d'effectifs. Ne mélange pas des unités marquées not_aggregatable ;
- utilise uniquement les tables/colonnes du schéma ;
- après toute union de tables ou de voies, crée obligatoirement une CTE de
  périmètre filtrant school_key sur les écoles demandées : les 16 school_key
  listés ci-dessus si aucune école précise n'est demandée. Les filtres de
  quelques branches ne suffisent pas. La campagne et tous les agrégats doivent
  lire cette CTE ou ses descendants, sans réintroduire les sources non filtrées.
  Conserve school_key jusqu'au calcul de COUNT(DISTINCT school_key). Si une
  contribution dépasse les 16 écoles du réseau, corrige les observations
  incluses, jamais le compteur par un plafond ou une limite artificiels ;
- conserve la même population, promotion et horizon au numérateur et au
  dénominateur. Distingue inscrits/diplômés/répondants/actifs et personnels
  de l'école. Une durée de recherche n'est pas une situation à six mois,
  « 6 mois ou plus » n'est pas strictement « plus de 6 mois », et création
  d'entreprise n'est pas un statut juridique autoentrepreneur ;
- déduplique les observations répétées avant toute somme d'effectifs ou calcul
  de taux, en conservant école, campagne, membre, voie, promotion/horizon et
  catégorie pertinents. N'utilise pas SUM(DISTINCT valeur) et ne cumule pas
  Total avec ses sous-totaux ou un effectif avec ses colonnes « dont » ;
- calcule les taux par 100.0 * SUM(numerateur) / NULLIF(SUM(denominateur), 0),
  sur les mêmes observations avec numérateur renseigné et dénominateur > 0.
  Zéro est une réponse valide, NULL reste inconnu. N'utilise pas AVG(taux_ecole)
  et ne laisse pas une jointure multiplier les observations ou les répondants ;
- si l'indicateur dispose d'un response_count associé, prends par défaut ce
  dénominateur du même champ, sauf définition différente explicitement demandée.
  Supprime les jointures et lectures de familles devenues inutiles : une famille
  sélectionnée n'est pas obligatoirement à utiliser dans le SQL. Apparier mesure
  et réponses à école/campagne/membre/voie/sous-groupe identiques avant les sommes.
  Ne remplace pas ces réponses par un total d'une population voisine ; respecte
  aussi les restrictions et exclusions propres au champ. Si le dénominateur
  demandé n'est pas disponible ou compatible, expose la limite du calcul ;
- inspecte tous les filtres intermédiaires : si le numérateur et le dénominateur
  occupent des lignes distinctes, conserve leurs deux catégories jusqu'au pivot
  au même grain école/campagne/membre/voie/horizon. Un WHERE limité aux lignes
  du numérateur rend impossible tout CASE ultérieur visant le dénominateur :
  corrige ce filtre en amont, pas le zéro final. Après déduplication, utilise
  MAX(CASE WHEN ... THEN valeur END) sans ELSE 0 pour apparier les catégories,
  puis filtre numerateur IS NOT NULL AND denominateur > 0 avant les sommes et
  le comptage des écoles. Ne remplace pas les effectifs manquants par zéro.
  Chaque horizon doit avoir ses propres couples valides et sa propre couverture ;
- si un CASE regroupe plusieurs catégories disjointes au numérateur, remplace
  le MAX par une somme des effectifs de ces catégories, après déduplication.
  MAX est adapté au pivot d'une catégorie unique, pas au cumul de tranches.
  N'additionne pas un total et ses composantes ; ne transforme pas une catégorie
  absente en zéro observé ;
- conserve l'horizon réel dans les alias : ne renomme pas une situation à
  6 mois ou 12–15 mois « à la sortie » pour répondre à une question ambiguë ;
- ne devine pas le sens d'un Total/Total_2 sans horizon explicite, et ne somme
  pas des salaires parce qu'un type de colonne erroné les présente comme count.
  Si les métadonnées se contredisent ou que la mesure exacte manque, expose
  les données pertinentes sans substituer une autre mesure à celle demandée ;
- retourne campagne, nombre d'écoles contributrices et, pour un taux, son
  numérateur et son dénominateur, pour permettre d'en expliquer la couverture ;
- un COUNT(DISTINCT school_key) dans un GROUP BY school_key est un compteur
  local, pas une couverture globale. Calcule la couverture sur toutes les
  observations valides dans une CTE séparée sans ce regroupement. Si le résultat
  reste ventilé par école, retourne school_key pour identifier chaque ligne ;
- si la question vise une propriété précisément nommée, le SQL doit filtrer
  le contenu sur cette propriété, y compris pour une simple liste. Exclure les
  réponses négatives ou vides ne suffit pas : une réponse affirmative portant
  sur un autre label, dispositif ou certification ne répond pas à la question.
  Les exemples cités dans l'intitulé de la rubrique ne constituent pas une
  preuve. Respecte les variantes typographiques du nom et distingue une
  attribution effective d'une candidature ou d'un objectif ;
- pour les demandes générales de labels de développement durable ou de
  « label DDRS », utilise toutes les déclarations positives de cti_v_5, sans
  filtrer sur un nom de label ni imposer la présence du sigle DD&RS/DDRS.
  Conserve les textes déclarés et les écoles correspondantes. Les réponses
  vides, négatives ou limitées à un projet/candidature restent exclues. Un
  filtre sur un label particulier nécessite une demande explicite excluant
  les autres labels ;
- pour une synthèse pondérée, conserve dans chaque branche UNION ALL une
  constante source_table et une constante voie qui identifie le
  couple valeur/poids. Avant le calcul, déduplique les observations identiques
  avec SELECT DISTINCT source_table, school_key, year, row_name, row_group,
  voie, valeur, poids. Les mêmes chiffres dans des voies ou membres
  différents restent des contributions distinctes. N'inclus pas les identifiants
  techniques de document dans cette clé et ne remplace pas simplement UNION ALL
  par UNION sans conserver ces identités ;
- filtre les écoles demandées et les couples valeur non NULL / poids > 0 avant
  de choisir la dernière campagne exploitable. Si une année est explicitement
  demandée, conserve cette année. Calcule ensuite
  1.0 * SUM(valeur * poids) / NULLIF(SUM(poids), 0) sur les données dédupliquées ;
- pour une question générale sur les salaires, combine FISE et FISA + FISEA
  au niveau Total (ou sous-groupe explicitement demandé), avec une valeur
  par catégorie de primes. Conserve la voie pour dédupliquer, puis regroupe
  seulement par year, row_name et row_group si nécessaire. Nomme la mesure
  moyenne_ponderee_des_medianes et arrondis uniquement le résultat final ;
- pour compter les entités qui possèdent/satisfont une propriété booléenne,
  catégorielle ou textuelle, ne confonds jamais IS NOT NULL, chaîne non vide
  ou NULLIF(TRIM(...),'') IS NOT NULL avec une réponse affirmative : inspecte
  le contenu et exclue explicitement les réponses négatives ou d'absence
  (« non », « aucun », « aucune », « néant », « sans ... », « pas de ... ») ;
- corrige la cause du problème en raisonnant à partir des libellés,
  types, unités, agrégations, dimensions, contexte et données réellement
  présentes dans SQLite ;
- une colonne peut exister dans le schéma mais être vide dans le périmètre :
  n'utilise pas une composante vide si une autre composante sémantiquement
  pertinente est réellement renseignée ;
- si l'erreur signale qu'une composante parallèle pondérée est renseignée mais
  absente du SQL, ajoute-la avec son response_count correspondant dans le même
  calcul pondéré ; ne remplace pas simplement une composante par une autre ;
- si l'erreur signale qu'une composante count parallèle est renseignée mais
  absente, ajoute toutes les populations pertinentes ; si le calcul utilise
  des response_count, ajoute aussi le response_count correspondant à chaque
  composante ;
- si l'erreur signale un mélange de sous-groupes, conserve le même sous-groupe
  dans tous les membres ; une colonne Total manquante ne doit jamais être
  remplacée par F, H ou une autre catégorie ;
- si l'erreur concerne la campagne implicite, sélectionne une seule année
  récente exploitable avant toute agrégation ;
- si l'erreur concerne plusieurs row_name d'une mesure pondérée, filtre la
  catégorie explicitement demandée ; si aucune n'est demandée, conserve
  row_name et retourne une valeur séparée par catégorie ;
- si l'erreur concerne une identité D*, ajoute un member_id ('d1', 'd2', ...)
  dans chaque branche ; sans école explicitement demandée, conserve aussi
  school_key et groupe/classe par le couple (school_key, member_id) ;
- une famille sélectionnée uniquement comme contexte descriptif ou pour aider à
  nommer une entité n'a pas à être ajoutée artificiellement au SQL si la mesure
  et l'identité structurelle sont déjà calculables avec une autre famille ;
- si l'erreur liste des colonnes absentes d'une table physique, corrige chaque
  branche séparément à partir des colonnes réellement disponibles indiquées ;
  n'applique pas automatiquement le schéma d'un membre D* à tous les autres ;
- si l'erreur concerne une addition de counts parallèles avec NULL, utilise
  COALESCE(colonne, 0) pour chacune des composantes ;
- si l'erreur signale des lignes identiques, corrige la jointure ou utilise
  DISTINCT/une agrégation appropriée sans perdre les catégories utiles ;
- un alias ne crée jamais une nouvelle dimension métier : ne renomme pas
  row_name, row_group ou school_key pour simuler l'entité demandée ;
- si le résultat était vide, revois notamment la campagne, les filtres,
  les colonnes choisies et la définition de la mesure, sans inventer de données ;
- n'ajoute aucune restriction absente de la question ;
- retourne uniquement le SQL.
""".strip()

    corrected = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Tu corriges une requête SQLite à partir du schéma "
                    "sémantique fourni. Retourne seulement le SQL."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_completion_tokens=3200,
    )

    return clean_sql(corrected)


# ---------------------------------------------------------------------------
# Réponse finale
# ---------------------------------------------------------------------------

def selected_semantics(selected_entries, schemas):
    lines = []

    for entry in selected_entries:
        context = candidate_context(entry, schemas)

        line = (
            f"- {entry['table']}: "
            f"{entry.get('sujet', '')}"
        )

        if context:
            line += f" | {context}"

        members = [
            table
            for table in entry.get("members", [])
            if table in schemas
        ]

        row_names = []
        row_groups = []

        for table in members:
            for value in get_dimension_values(
                schemas[table],
                "row_name",
            ):
                if value not in row_names:
                    row_names.append(value)

            for value in get_dimension_values(
                schemas[table],
                "row_group",
            ):
                if value not in row_groups:
                    row_groups.append(value)

        if row_names:
            shown = row_names[:MAX_DIMENSION_VALUES_IN_PROMPT]
            line += (
                " | row_name observé="
                + " | ".join(shown)
            )

        if row_groups:
            shown = row_groups[:MAX_DIMENSION_VALUES_IN_PROMPT]
            line += (
                " | row_group observé="
                + " | ".join(shown)
            )

        aggregations = []
        types = []

        for table in members:
            for column in parse_business_columns(
                schemas.get(table, "")
            ):
                aggregation = column.get("aggregation", "")
                column_type = column.get("type", "")

                if aggregation and aggregation not in aggregations:
                    aggregations.append(aggregation)

                if column_type and column_type not in types:
                    types.append(column_type)

        if aggregations:
            line += (
                " | agrégations="
                + " | ".join(aggregations)
            )

        if types:
            line += (
                " | types="
                + " | ".join(types)
            )

        lines.append(line)

    return "\n".join(lines)


def generate_final_answer(
    llm,
    question,
    columns,
    rows,
    selected_entries,
    schemas,
    member_label_context="",
    coverage_context="",
    sql="",
):
    safe_rows = rows[:MAX_FINAL_ROWS]
    requested_years = explicit_years_in_question(question)
    explicit_year_context = (
        ", ".join(str(year) for year in requested_years)
        if requested_years
        else "aucune"
    )

    # Métadonnées de provenance uniquement : aucune interprétation métier
    # de la question et aucun appel LLM supplémentaire.
    source_lines = []
    for table in sorted(referenced_physical_tables(sql)):
        schema = schemas.get(table, "")
        if not schema:
            continue
        document = table.split("_", 1)[0].upper()
        code = schema_header_value(schema, "CODE")
        section = schema_header_value(schema, "SECTION")
        subject = schema_header_value(schema, "SUJET")
        source_lines.append(
            f"- Document {document} ; rubrique {code} ; "
            f"partie : {section} ; sujet : {subject}"
        )
    source_context = "\n".join(source_lines) or "Provenance non disponible."

    prompt = f"""
QUESTION:
{question}

ANNÉE(S) EXPLICITEMENT DEMANDÉE(S) DANS LA QUESTION:
{explicit_year_context}

DONNÉES SÉLECTIONNÉES:
{selected_semantics(selected_entries, schemas)}

DOCUMENTS ET PARTIES RÉFÉRENCÉS PAR LE SQL:
{source_context}

{member_label_context}

{coverage_context}

COLONNES DU RÉSULTAT:
{json.dumps(columns, ensure_ascii=False)}

RÉSULTAT SQLITE:
{json.dumps(safe_rows, ensure_ascii=False, default=str)}

Réponds en français uniquement à partir du résultat SQLite et de la
sémantique fournie.

- n'invente aucune donnée ;
- conserve fidèlement les valeurs et unités ;
- sauf ventilation explicitement demandée, la question vise une synthèse globale
  du réseau. Si le résultat ne contient que des chiffres par école, précise que
  le calcul global demandé n'a pas été obtenu ; ne les moyenne pas mentalement ;
- des valeurs de type percentage/unité percent sont des pourcentages sur 100,
  jamais des nombres de personnes. Si le résultat additionne des pourcentages
  d'écoles, signale que cette somme ne constitue ni un effectif ni une proportion
  du réseau. Un ratio de sommes de parts non pondérées ne représente pas le ratio
  global des personnes. Ne présente pas ce résultat comme valide et ne fabrique
  pas de correction sans les effectifs ou dénominateurs correspondants ;
- termine par une courte mention « Source : document … — rubrique … : … »
  fondée exclusivement sur le bloc DOCUMENTS ET PARTIES RÉFÉRENCÉS PAR LE SQL.
  Donne le type de document (CTI ou DEFI), le code de la rubrique et son sujet
  en français clair ; ajoute l'intitulé de la partie lorsque cela aide à situer
  la mesure. Si plusieurs rubriques sont utilisées, cite-les, en regroupant
  les rubriques homologues D1, D2, etc. pour rester concis. Ne présente pas
  uniquement un identifiant technique comme defi_vii_1 ou cti_viii_d1_1.
  N'invente aucun nom de fichier, numéro de page ou provenance absents de ce
  bloc. Un millésime contenu dans un titre commun de partie ne date pas à lui
  seul les observations retournées : ne le recopie pas comme année de la mesure
  sans correspondance vérifiée. Cite alors le titre sans son millésime et
  distingue la campagne SQL de la période réelle. Si la provenance n'est pas
  disponible, indique-le brièvement ;
- pour toute synthèse du réseau, indique la couverture lorsqu'elle est
  disponible : nombre d'écoles contributrices sur les 16 du réseau, et portée
  des voies/membres exploitables. Un sous-ensemble ne représente pas un total
  exhaustif du réseau. Une école sans données n'a pas forcément une valeur zéro ;
- distingue couverture des écoles et couverture des voies/membres : ne présente
  pas « 16 écoles sur 16 » comme la preuve d'une synthèse toutes voies. Pour une
  question générale, le calcul doit inclure les contributions compatibles FISE
  et FISA/FISEA sur tous les membres disponibles ; si le résultat documente une
  restriction, indique-la plutôt que d'affirmer une exhaustivité non établie ;
- n'interprète jamais un compteur local de 1 répété sur plusieurs lignes comme
  « 1 école contributrice sur 16 ». Si les school_key distincts sont fournis,
  la liste permet d'identifier les écoles présentes ; sinon ne devine pas le
  nombre d'écoles à partir de lignes anonymes et indique la couverture globale
  non établie. Ne reprends pas non plus un millésime de titre commun de partie
  comme date des observations sans preuve de correspondance ;
- si le résultat annonce plus de 16 écoles contributrices pour le réseau
  défini ici, signale une incohérence de périmètre SQL. Ne présente pas le taux
  ou le total associé comme un chiffre valide du réseau, et n'écris jamais
  « 17 écoles sur 16 » comme une couverture normale. Ne ramène pas simplement
  le compteur à 16 et n'invente pas de valeurs corrigées : le SQL doit être
  recalculé après correction du filtre ;
- pour un taux, nomme le numérateur, le dénominateur et la population réellement
  mesurés, au moins brièvement (par exemple « parmi les répondants à ce champ »).
  Ne transforme pas une part de répondants en taux portant sur tous les diplômés
  ni en taux net d'emploi parmi les actifs. Une couverture de 16 écoles ne
  garantit pas que tous les diplômés, membres ou voies sont renseignés ;
- lorsqu'un taux utilise le nombre de réponses propre à un champ, présente-le
  comme une proportion parmi les répondants à ce champ, en conservant ses
  restrictions et exclusions. Ne le renomme pas comme un taux de tous les
  diplômés ou de toute une population voisine ;
- un dénominateur nul dans le résultat SQL ne prouve pas que les données sources
  contiennent zéro répondant : un filtre ou un appariement peut en être la cause.
  Sans preuve sur la source, dis « le dénominateur retourné par la requête est
  nul ; ce résultat ne permet pas de calculer le taux », sans attribuer ce zéro
  aux données d'origine. Ne déduis pas une couverture valide du taux à partir
  d'un simple comptage d'écoles ayant uniquement un numérateur ;
- distingue un délai pour trouver le premier emploi et le statut à une date
  d'observation. Conserve les bornes de la donnée : « 6 mois ou plus » signifie
  au moins six mois. Si la question demande strictement plus de six mois, précise
  que cette tranche ne permet pas de calculer exactement ce seuil ;
- si la question ne fixe pas d'horizon d'observation, annonce celui de la source
  choisie sans le présenter comme imposé par l'utilisateur. Ne transforme pas
  une observation en janvier en observation à six mois. Si la question fixe un
  horizon que le résultat ne respecte pas, explique le décalage même si la
  couverture du résultat est meilleure ;
- distingue effectifs d'apprentis inscrits, diplômés apprentis et contrats.
  Création d'entreprise, statut étudiant entrepreneur et régime juridique
  autoentrepreneur ne sont pas interchangeables. Si le concept exact n'est pas
  disponible, dis-le même si le SQL fournit un indicateur voisin ;
- si les métadonnées sont ambiguës ou contradictoires, indique la limite du
  calcul au lieu de certifier le résultat. Ne corrige pas mentalement des
  cellules extraites ou des totaux sans données justificatives ;
- convention de cette application : une demande générale de labels de
  développement durable, y compris « quelles écoles ont un label DDRS »,
  couvre toutes les déclarations positives de la rubrique cti_v_5. Présente
  toutes les écoles et les textes déclarés retournés par le SQL ; ne refiltre
  pas cette liste pour ne garder que les mentions « Label DD&RS ». Présente
  ces données comme des déclarations dans la rubrique développement durable,
  sans renommer les dispositifs ni affirmer qu'ils portent tous le label
  officiel DD&RS. Une restriction à ce label seul doit être explicitement
  demandée à l'exclusion des autres ;
- pour une liste d'entités satisfaisant une propriété, vérifie que chaque
  ligne retenue atteste précisément la propriété demandée dans ses valeurs,
  et pas seulement dans le libellé général de la rubrique. Ne transforme pas
  une autre propriété, une intention ou une candidature en attribution acquise ;
- si le résultat SQL contient des lignes qui ne répondent pas au critère de
  la question, ne les inclus pas comme réponses positives et ne masque pas
  ce décalage : précise brièvement que le résultat SQL est plus large que le
  critère demandé et que seules les lignes qui l'attestent sont retenues.
  Si le texte ne permet pas de trancher, indique l'incertitude plutôt que
  d'affirmer une attribution ou une absence. Pour une question demandant une
  liste complète, cite toutes les entités répondant au critère parmi les lignes
  fournies ; ne sélectionne pas arbitrairement quelques exemples ;
- IMPORTANT : si la sémantique sélectionnée contient
  weighted_average_by_response_count, le résultat synthétique n'est PAS une
  médiane statistique globale. Dans la réponse finale, nomme explicitement la
  mesure « moyenne pondérée des salaires médians » ; n'écris jamais simplement
  « salaire médian » pour cette valeur synthétique ;
- pour toute valeur monétaire, arrondis à 2 décimales sauf si la question
  demande explicitement davantage de précision ;
- pour un pourcentage, arrondis à 2 décimales sauf si la question demande
  explicitement davantage de précision ;
- la colonne SQL year représente l'année/campagne utilisée pour sélectionner
  les données. Si la question ne contient AUCUNE année explicite, ne transforme
  pas automatiquement cette campagne technique en affirmation « en YYYY » sur
  la période réelle de la mesure. Donne d'abord la valeur demandée ; si l'année
  est utile, appelle-la explicitement « campagne YYYY » ;
- si la question contient une année explicite, indique cette année normalement
  puisque le SQL a déjà été validé pour la respecter ;
- le filtre school_key définit l'école concernée. Une mention comme « en France »
  dans le sujet ou le contexte d'un indicateur peut décrire le lieu de l'emploi,
  du salaire ou de la mesure ; elle ne remplace jamais l'école filtrée par
  school_key. Si le SQL filtre MARSEILLE, la réponse concerne Marseille ;
- ne réinterprète jamais une dimension retournée : school_key reste une école,
  year reste une année/campagne, et row_name/row_group gardent le sens indiqué
  par leurs valeurs observées. Un row_name « Avec prime » n'est par exemple
  qu'une catégorie de row_name, pas une autre entité inventée ;
- si la question demande d'identifier ou classer une entité mais que le
  résultat SQLite ne contient aucun identifiant/libellé réellement associé à
  cette entité, indique clairement que le résultat ne permet pas de l'identifier
  au lieu de répondre avec une autre dimension ;
- si un bloc « LIBELLÉS HUMAINS DES MEMBRES D* » est fourni, utilise le libellé
  humain correspondant exactement au couple school_key + member_id ; ne
  présente jamais d1/d2/d3 comme le nom final d'une spécialité et ne suppose
  jamais qu'un même dN désigne la même formation dans deux écoles différentes ;
- si un bloc « COUVERTURE PARTIELLE DE FAMILLES D* » est fourni, ne présente
  jamais la valeur comme un total exhaustif de tous les membres. Qualifie
  explicitement la portée, par exemple en indiquant que le résultat porte sur
  les membres/spécialités pour lesquels des colonnes compatibles et des données
  exploitables sont disponibles. Ne transforme pas cette limite structurelle en
  estimation des membres exclus ;
- si plusieurs lignes sont utiles, résume-les clairement ;
- si le résultat ne répond pas réellement à la question, dis-le ;
- reste concis.
""".strip()

    return llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Tu formules la réponse finale d'un système "
                    "question-réponse sur des données structurées."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_completion_tokens=500,
    ).strip()


# ---------------------------------------------------------------------------
# Moteur principal
# ---------------------------------------------------------------------------

class DataQueryEngine:

    def __init__(
        self,
        llm,
        db_path="cti_defi_sections.sqlite",
        schema_path="semantic_schema.txt",
        top_k=30,
        max_attempts=3,
        **_ignored_compatibility_options,
    ):
        self.llm = llm
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path)
        self.top_k = top_k
        self.max_attempts = max_attempts

        self.schemas = load_schema_blocks(
            self.schema_path
        )

        # L'index lexical BM25 dépend uniquement du semantic_schema.
        # Il est donc construit une seule fois au démarrage du moteur.
        self.bm25_index = build_bm25_schema_index(
            self.schemas
        )

    def ask(self, question):
        raw_candidates = search_tables(
            question,
            top_k=self.top_k,
        )

        embedding_candidates = normalize_candidates(
            raw_candidates,
            self.schemas,
        )

        bm25_candidates = bm25_schema_candidates(
            question,
            self.bm25_index,
            top_k=BM25_RECALL_K,
        )

        candidates = merge_hybrid_candidates(
            embedding_candidates,
            bm25_candidates,
            self.schemas,
        )

        if not candidates:
            raise ValueError(
                "Aucune famille candidate n'a été trouvée."
            )

        routing_connection = sqlite3.connect(self.db_path)
        try:
            routing_connection.execute("PRAGMA query_only = ON")
            candidate_coverage = build_candidate_coverage(
                routing_connection, candidates, self.schemas,
            )
        finally:
            routing_connection.close()

        selected_entries = route_tables(
            self.llm,
            question,
            candidates,
            self.schemas,
            coverage=candidate_coverage,
        )

        selected_entries = validate_or_correct_entity_route(
            self.llm,
            question,
            candidates,
            selected_entries,
            self.schemas,
        )

        physical_tables = expand_families(
            selected_entries
        )

        if not physical_tables:
            raise ValueError(
                "Aucune table physique n'est associée "
                "aux familles sélectionnées."
            )

        connection = sqlite3.connect(self.db_path)

        try:
            # Barrière de sécurité SQLite.
            connection.execute("PRAGMA query_only = ON")

            initial_data_profile = build_question_data_profile(
                connection,
                question,
                physical_tables,
                self.schemas,
            )

            sql, _ = generate_sql(
                self.llm,
                question,
                selected_entries,
                self.schemas,
                data_profile=initial_data_profile,
            )

            rows = None
            columns = None
            valid = False
            error = None
            runtime_profile = ""
            previous_result = ""

            for attempt in range(self.max_attempts):
                sql = canonicalize_dimension_literals(
                    sql,
                    physical_tables,
                    self.schemas,
                )

                valid, error = validate_columns_per_physical_table(
                    connection,
                    sql,
                    physical_tables,
                )

                if valid:
                    valid, error = validate_explicit_year_scope(
                        question,
                        sql,
                    )

                if valid:
                    valid, error = validate_school_scope(
                        question,
                        sql,
                    )

                if valid:
                    valid, error = validate_implicit_year_scope(
                        question,
                        sql,
                        physical_tables,
                        self.schemas,
                    )

                if valid:
                    valid, error = validate_alias_semantic_shadowing(
                        sql,
                        physical_tables,
                        self.schemas,
                    )

                if valid:
                    valid, error = validate_null_safe_parallel_addition(
                        sql,
                        physical_tables,
                        self.schemas,
                    )

                if valid:
                    valid, error = validate_virtual_family_coverage(
                        connection,
                        question,
                        sql,
                        selected_entries,
                        self.schemas,
                    )

                if valid:
                    valid, error = validate_aggregation_contracts(
                        sql,
                        physical_tables,
                        self.schemas,
                    )

                if valid:
                    valid, error = validate_sql(
                        connection,
                        sql,
                        physical_tables,
                        self.schemas,
                    )

                if valid:
                    try:
                        cursor = connection.execute(sql)
                        rows = cursor.fetchall()
                        columns = (
                            [
                                column[0]
                                for column in cursor.description
                            ]
                            if cursor.description
                            else []
                        )
                    except sqlite3.Error as exc:
                        valid = False
                        error = str(exc)

                if valid:
                    valid, error = validate_weighted_component_coverage(
                        connection,
                        question,
                        sql,
                        physical_tables,
                        self.schemas,
                        result_columns=columns,
                        result_rows=rows,
                    )

                if valid:
                    valid, error = validate_parallel_count_components(
                        connection,
                        question,
                        sql,
                        physical_tables,
                        self.schemas,
                        result_columns=columns,
                        result_rows=rows,
                    )

                if valid:
                    valid, error = validate_weighted_dimension_scope(
                        question,
                        sql,
                        physical_tables,
                        self.schemas,
                        columns,
                        rows,
                    )

                if valid:
                    valid, error = validate_virtual_member_identity(
                        question,
                        selected_entries,
                        columns,
                        rows,
                        sql=sql,
                    )

                if valid:
                    valid, error = validate_duplicate_result_rows(
                        columns,
                        rows,
                    )

                if valid and not result_is_usable(
                    rows,
                    columns,
                ):
                    valid = False
                    error = (
                        "La requête est valide mais ne retourne aucune "
                        "valeur métier exploitable. Vérifie la campagne, "
                        "les filtres, les colonnes choisies et la définition "
                        "de la mesure."
                    )

                    previous_result = failed_result_context(
                        columns,
                        rows,
                    )

                    runtime_profile = build_runtime_data_profile(
                        connection,
                        physical_tables,
                        self.schemas,
                        sql,
                    )

                if valid:
                    break

                if attempt >= self.max_attempts - 1:
                    break

                sql = correct_sql(
                    self.llm,
                    question,
                    sql,
                    error,
                    selected_entries,
                    self.schemas,
                    runtime_profile=runtime_profile,
                    previous_result=previous_result,
                )

            if not valid:
                raise ValueError(
                    "Impossible d'obtenir un SQL exploitable après "
                    f"{self.max_attempts} tentative(s) : {error}\n\n"
                    f"SQL final:\n{sql}"
                )

            member_label_context = build_virtual_member_label_context(
                connection,
                question,
                selected_entries,
                columns,
                rows,
            )

            coverage_context = build_virtual_family_coverage_context(
                connection,
                question,
                sql,
                selected_entries,
                self.schemas,
            )

            answer = generate_final_answer(
                self.llm,
                question,
                columns,
                rows,
                selected_entries,
                self.schemas,
                member_label_context=member_label_context,
                coverage_context=coverage_context,
                sql=sql,
            )

            return {
                "answer": answer,
                "sql": sql,
                "columns": columns,
                "rows": rows,
                "families": [
                    entry["table"]
                    for entry in selected_entries
                ],
                "physical_tables": physical_tables,
                "candidates": candidates,
                "coverage_context": coverage_context,
            }

        finally:
            connection.close()
