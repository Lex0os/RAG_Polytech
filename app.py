from pathlib import Path
import subprocess
import sys
import streamlit as st

from llm.groq_provider import GroqProvider
from llm.openai_provider import OpenAIProvider
from query_engine import DataQueryEngine


PROJECT_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_DIR / "documents_brut"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Assistant Polytech",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Assistant Polytech")
st.caption("Question → LLM → SQLite → Python → LLM")


# -----------------------
# LLM + moteur
# -----------------------

@st.cache_resource
# Version Groq
#def get_engine():
#    llm = GroqProvider(
#        api_key=st.secrets["GROQ_API_KEY"],
#        model="openai/gpt-oss-20b",
#    )
#
#    return DataQueryEngine(
#        llm=llm,
#       db_path=Path("cti_defi_sections.sqlite"),
#       schema_path=Path("semantic_schema.txt"),
#        top_k=10,
#        max_attempts=3,
#    )
#Version OpenAI
def get_engine():
    llm = OpenAIProvider(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-5.6-luna",
        reasoning_effort="none",
    )

    return DataQueryEngine(
        llm=llm,
        db_path=Path("cti_defi_sections.sqlite"),
        schema_path=Path("semantic_schema.txt"),
        top_k=30,
        max_attempts=3,
    )

engine = get_engine()


# -----------------------
# Historique
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------
# Question utilisateur
# -----------------------

question = st.chat_input(
    "Posez une question sur les données..."
)


if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche et génération de la requête..."):
                result = engine.ask(question)

            st.markdown(result["answer"])

            with st.expander("Détails techniques"):
                st.write(
                    "Famille(s) sélectionnée(s) :",
                    ", ".join(result["families"]),
                )
                st.write(
                    "Tables physiques :",
                    ", ".join(result["physical_tables"]),
                )
                st.code(result["sql"], language="sql")

                st.write("Résultat brut SQLite :")
                st.dataframe(
                    [dict(zip(result["columns"], row)) for row in result["rows"]],
                    use_container_width=True,
                )
            with st.expander("Candidats retrieval"):
                for i, candidate in enumerate(result["candidates"], start=1):
                    embedding_score = candidate.get("score")
                    lexical_score = candidate.get("lexical_score")
                    source = candidate.get("retrieval_source", "embedding")

                    if embedding_score is not None:
                        score_text = f"embedding: {embedding_score:.3f}"
                    elif lexical_score is not None:
                        score_text = f"lexical: {lexical_score:.3f}"
                    else:
                        score_text = "score: n/a"

                    st.write(
                        f"{i}. {candidate['table']} "
                        f"— {score_text}"
                    )

                    st.caption(
                        f"{candidate.get('sujet', '')} "
                        f"· source: {source}"
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
            })

        except Exception as e:
            st.error(str(e))
#
# Extraction et indexation des documents PDF
#
with st.sidebar:
    st.header("Ajouter des documents")

    uploaded_files = st.file_uploader(
        "Ajouter des fichiers PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} fichier(s) sélectionné(s)")

        if st.button("Extraire et indexer"):
            try:
                # 1. Sauvegarde des PDF
                with st.status("Traitement des documents...", expanded=True) as status:

                    st.write("Sauvegarde des fichiers...")

                    for uploaded_file in uploaded_files:
                        destination = UPLOAD_DIR / uploaded_file.name

                        with open(destination, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # 2. Extraction PDF -> SQLite
                    st.write("Extraction des données...")

                    subprocess.run(
                        [
                            sys.executable,
                            str(PROJECT_DIR / "extracteur_sections_cti_defi.py"),
                            str(UPLOAD_DIR),
                            "--sqlite",
                            str(PROJECT_DIR / "cti_defi_sections.sqlite"),
                            "--replace",
                        ],
                        cwd=PROJECT_DIR,
                        check=True,
)

                    # 3. Régénération du schéma sémantique
                    st.write("Génération du schéma sémantique...")

                    subprocess.run(
                        [
                            sys.executable,
                            "generate_semantic_schema.py",
                        ],
                        check=True,
                    )

                    # 4. Régénération des embeddings
                    st.write("Génération des embeddings...")

                    subprocess.run(
                        [
                            sys.executable,
                            "embedding.py",
                        ],
                        check=True,
                    )

                    status.update(
                        label="Documents ajoutés avec succès",
                        state="complete",
                    )

                st.success(
                    "Extraction et indexation terminées. "
                    "Les nouveaux documents peuvent maintenant être interrogés."
                )

            except subprocess.CalledProcessError as e:
                st.error(
                    f"Une erreur est survenue pendant le traitement : {e}"
                )