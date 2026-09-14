"""
Quick manual test of the running FastAPI server.
Run the server first (uvicorn main:app --reload), then run this script
in a separate terminal.
"""
import requests

BASE_URL = "http://127.0.0.1:8000"

print("--- Valid input ---")
r = requests.post(f"{BASE_URL}/predict", json={"text": "I do not recognize this payment."})
print(r.status_code, r.json())

print("\n--- Valid input 2 ---")
r = requests.post(f"{BASE_URL}/predict", json={"text": "I want a new account"})
print(r.status_code, r.json())

print("\n--- Invalid: wrong type ---")
r = requests.post(f"{BASE_URL}/predict", json={"text": 12345})
print(r.status_code, r.json())

print("\n--- Invalid: missing field ---")
r = requests.post(f"{BASE_URL}/predict", json={})
print(r.status_code, r.json())

print("\n--- Invalid: empty string ---")
r = requests.post(f"{BASE_URL}/predict", json={"text": ""})
print(r.status_code, r.json())
