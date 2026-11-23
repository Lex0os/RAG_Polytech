from pathlib import Path
import subprocess
import sys
import streamlit as st

from llm.groq_provider import GroqProvider
from llm.llama_provider import LlamaProvider
from llm.openai_provider import OpenAIProvider
from query_engine import DataQueryEngine


PROJECT_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_DIR / "documents_brut"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def afficher_pdf(nom_pdf, cle):
    fichier = Path(nom_pdf)
    if fichier.name != nom_pdf or fichier.suffix.lower() != ".pdf":
        st.warning("Nom de document invalide.")
        return

    try:
        chemin_pdf = (UPLOAD_DIR / fichier).resolve(strict=True)
        if (
            chemin_pdf.parent != UPLOAD_DIR.resolve()
            or chemin_pdf.suffix.lower() != ".pdf"
        ):
            st.warning("Emplacement du document invalide.")
            return
        contenu_pdf = chemin_pdf.read_bytes()
    except (OSError, RuntimeError):
        st.warning(f"Document PDF indisponible : {nom_pdf}")
        return

    st.pdf(contenu_pdf, height=650, key=f"{cle}_lecteur", alt=nom_pdf)
    st.download_button(
        "Télécharger ce PDF",
        data=contenu_pdf,
        file_name=nom_pdf,
        mime="application/pdf",
        key=f"{cle}_telechargement",
        on_click="ignore",
    )


def afficher_sources_pdf(sources, cle):
    with st.expander("Sources PDF de cette réponse"):
        if not sources:
            st.info("Aucun document source enregistré pour cette réponse.")
            return

        for nom_pdf in sources:
            st.write(nom_pdf)

        document = st.selectbox(
            "Choisir une source à consulter",
            [None] + sources,
            format_func=lambda nom: nom or "Choisir un PDF",
            key=f"{cle}_selection",
        )
        if document is not None:
            afficher_pdf(document, cle)


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

@st.cache_resource(scope="session")
def get_engine(provider, model, api_key, base_url):
    if provider == "OpenAI":
        options = {}
        if model != "gpt-5.6-luna":
            options = {
                "reasoning_effort": None,
                "verbosity": None,
            }
        llm = OpenAIProvider(api_key=api_key, model=model, **options)
    elif provider == "Groq":
        llm = GroqProvider(api_key=api_key, model=model)
    else:
        llm = LlamaProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    return DataQueryEngine(
        llm=llm,
        db_path=Path("cti_defi_sections.sqlite"),
        schema_path=Path("semantic_schema.txt"),
        top_k=30,
        max_attempts=3,
    )

with st.sidebar:
    st.header("Modèle de langage")

    default_models = {
        "OpenAI": "gpt-5.6-luna",
        "Groq": "openai/gpt-oss-20b",
        "Llama": "local",
    }
    provider = st.selectbox(
        "Fournisseur",
        list(default_models),
        key="llm_provider",
    )
    model = st.text_input(
        "Modèle",
        value=default_models[provider],
        key=f"llm_model_{provider}",
        help="Identifiant du modèle à utiliser chez ce fournisseur.",
    ).strip()

    base_url = ""
    if provider == "Llama":
        base_url = st.text_input(
            "Adresse du serveur",
            value="http://127.0.0.1:8080/v1",
            key="llm_base_url",
            help="Adresse du serveur Llama compatible avec l'API OpenAI.",
        ).strip()

    try:
        configured_api_key = st.secrets.get(f"{provider.upper()}_API_KEY", "")
    except FileNotFoundError:
        configured_api_key = ""

    api_key = st.text_input(
        "Clé API",
        type="password",
        key=f"llm_api_key_{provider}",
        placeholder="Clé configurée" if configured_api_key else "Saisir une clé",
        help=(
            "Laisser vide pour utiliser la clé configurée dans secrets.toml. "
            "Une clé est facultative pour un serveur Llama local."
        ),
    ).strip() or configured_api_key
    if provider == "Llama" and not api_key:
        api_key = "local"

    configuration_error = None
    if not model:
        configuration_error = "Renseignez le modèle à utiliser."
    elif not api_key:
        configuration_error = f"Renseignez une clé API pour {provider}."
    elif provider == "Llama" and not base_url.startswith(("http://", "https://")):
        configuration_error = "Renseignez une adresse HTTP ou HTTPS pour le serveur Llama."

    if configuration_error:
        st.warning(configuration_error)


# -----------------------
# Historique
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for index_message, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            afficher_sources_pdf(
                message.get("source_files", []),
                f"message_{index_message}",
            )


# -----------------------
# Question utilisateur
# -----------------------

question = st.chat_input(
    "Posez une question sur les données...",
    disabled=configuration_error is not None,
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
                engine = get_engine(provider, model, api_key, base_url)
                result = engine.ask(question)

            sources = result.get("source_files", [])
            st.markdown(result["answer"])
            afficher_sources_pdf(
                sources,
                f"message_{len(st.session_state.messages)}",
            )

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
                "source_files": sources,
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

with st.sidebar:
    st.header("Consulter les documents")
    noms_pdf = sorted(
        fichier.name
        for fichier in UPLOAD_DIR.iterdir()
        if fichier.is_file() and fichier.suffix.lower() == ".pdf"
    )
    document_libre = st.selectbox(
        "Tous les PDF disponibles",
        [None] + noms_pdf,
        format_func=lambda nom: nom or "Choisir un document",
        key="document_libre_selection",
    )

if document_libre is not None:
    with st.expander(f"Document : {document_libre}", expanded=True):
        afficher_pdf(document_libre, "document_libre")
