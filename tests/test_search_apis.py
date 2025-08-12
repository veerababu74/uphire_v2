import requests
import json


def test_vector_search():
    url = "http://127.0.0.1:8000/rag/vector-similarity-search"
    payload = {"query": "python developer", "limit": 5}

    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_llm_context_search():
    url = "http://127.0.0.1:8000/rag/llm-context-search"
    payload = {"query": "experienced python developer", "context_size": 5}

    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Vector Similarity Search...")
    test_vector_search()

    print("\nTesting LLM Context Search...")
    test_llm_context_search()
