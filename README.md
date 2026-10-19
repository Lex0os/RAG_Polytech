Ce repo contient d'abord le resultat de mon stage de 4A. Il contiendra ensuite le TAI de 5eme année.
Il faut un fournisseur d'API pour faire fonctionner la partie LLM. Il faut aussi un modèle d'embedding qui tourne en local.
Dans le dossier LLM il existe deux possiblités Groq qui fournit plein de modèles différents ou OpenAI que j'ai utilisé.
Pour le modèle d'embedding j'utilise personellement : Qwen3-Embedding-0.6B-Q8_0.gguf.
Pour faire fonctionner l'API il faut créer un dossier .streamlit et mettre les clé dans un fichier secrets.toml