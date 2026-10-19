import json

from groq import Groq


class GroqProvider:

    def __init__(self, api_key, model="openai/gpt-oss-20b"):
        self.client = Groq(api_key=api_key)
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
            "max_completion_tokens": max_completion_tokens,
            "include_reasoning": False,
            "reasoning_effort": "low",
        }

        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        return response.choices[0].message.content or ""

    def chat_json(self, messages, schema=None, max_completion_tokens=4096):
        if schema is None:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": schema,
                },
            }

        return json.loads(
            self.chat(
                messages,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
            )
        )
