import json

from openai import OpenAI


class LlamaProvider:
    """Provider Llama compatible avec l'interface de query_engine.py.

    Utilise un serveur compatible avec l'API Chat Completions d'OpenAI.
    Par défaut, le serveur llama.cpp est attendu sur le port 8080.
    Le modèle doit être chargé côté serveur ; ``model`` désigne son alias.
    """

    def __init__(
        self,
        api_key="local",
        model="local",
        base_url="http://127.0.0.1:8080/v1",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(
        self,
        messages,
        response_format=None,
        max_completion_tokens=1024,
    ):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_completion_tokens,
        }

        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        return response.choices[0].message.content or ""

    def chat_json(
        self,
        messages,
        schema=None,
        max_completion_tokens=4096,
    ):
        if schema is None:
            response_format = {
                "type": "json_object",
            }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": schema,
                },
            }

        content = self.chat(
            messages,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
        )

        if not content.strip():
            raise RuntimeError("Llama a retourné une réponse vide.")

        return json.loads(content)