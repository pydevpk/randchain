from typing import Any


def ollama_chat(prompt: str, model_id: str, query: Any):
    import ollama
    response = ollama.chat(
        model=model_id,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )

    return response["message"]["content"]