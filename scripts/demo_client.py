from __future__ import annotations

import json

import httpx


API_URL = "http://127.0.0.1:8000/ask"


def main() -> None:
    question = "Can I return an item after 40 days?"
    payload = {"question": question}

    response = httpx.post(API_URL, json=payload, timeout=15.0)
    response.raise_for_status()

    print("Question:")
    print(question)
    print("\nResponse JSON:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
