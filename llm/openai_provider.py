import json

from openai import OpenAI


class OpenAIProvider:
    """Provider OpenAI compatible avec l'interface attendue par query_engine.py.

    Le moteur continue d'appeler uniquement :
    - chat(...)
    - chat_json(...)

    Les appels utilisent la Responses API d'OpenAI.
    """

    def __init__(
        self,
        api_key,
        model="gpt-5.6-luna",
        reasoning_effort="none",
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def _responses_text_config(response_format=None):
        text_config = {
            "verbosity": "low",
        }

        if response_format is None:
            return text_config

        response_type = response_format.get("type")

        if response_type == "json_schema":
            json_schema = response_format.get("json_schema", {})
            text_config["format"] = {
                "type": "json_schema",
                "name": json_schema.get("name", "response"),
                "schema": json_schema.get("schema", {}),
                "strict": json_schema.get("strict", True),
            }

        elif response_type == "json_object":
            text_config["format"] = {
                "type": "json_object",
            }

        return text_config

    def chat(
        self,
        messages,
        response_format=None,
        max_completion_tokens=1024,
    ):
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            max_output_tokens=max_completion_tokens,
            reasoning={
                "effort": self.reasoning_effort,
            },
            text=self._responses_text_config(response_format),
            store=False,
        )

        return response.output_text or ""

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
            raise RuntimeError("OpenAI a retourné une réponse vide.")

        return json.loads(content)
