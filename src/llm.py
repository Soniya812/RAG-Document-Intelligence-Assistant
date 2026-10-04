import os
from typing import List

import openai


class LLM:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not provided")

        self.api_key = key
        self.model = model

        if hasattr(openai, "OpenAI"):
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            openai.api_key = self.api_key
            self.client = openai

    def generate_answer(self, question: str, contexts: List[str], max_tokens: int = 300) -> str:
        valid_contexts = [context for context in contexts if context and str(context).strip()]
        if not valid_contexts:
            return "I don't know based on the provided context."

        prompt = self.build_prompt(question, valid_contexts)

        try:
            if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()

            if hasattr(openai, "ChatCompletion"):
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return resp["choices"][0]["message"]["content"].strip()

            raise RuntimeError("OpenAI client is not configured for chat completions.")
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    @staticmethod
    def build_prompt(question: str, contexts: List[str]) -> str:
        header = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
            "If the answer is not contained in the context, say you don't know.\n\n"
        )
        context_text = "\n\n---\n\n".join(contexts)
        prompt = f"{header}CONTEXT:\n{context_text}\n\nQUESTION: {question}\n\nAnswer:"
        return prompt
