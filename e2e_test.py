import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"
s = requests.Session()

def run_test():
    print("1. Registering User...")
    user_data = {
        "username": "e2etestuser",
        "email": "e2e@test.com",
        "password": "testpassword",
        "full_name": "E2E Tester"
    }
    r = s.post(f"{BASE_URL}/auth/register", json=user_data)
    if r.status_code == 400 and ('already exists' in r.text.lower() or 'already registered' in r.text.lower()):
        print(" User already exists.")
    else:
        r.raise_for_status()
        print(" Registered successfully.")

    print("2. Logging in...")
    login_data = {
        "username": "e2etestuser",
        "password": "testpassword"
    }
    r = s.post(f"{BASE_URL}/auth/login", json=login_data)
    r.raise_for_status()
    token = r.json().get('access_token')
    s.headers.update({"Authorization": f"Bearer {token}"})
    print(" Logged in successfully.")

    print("3. Uploading Document...")
    files = {'file': ('test_policy.txt', "This is a test privacy policy that states we collect your data and sell it to third parties.", 'text/plain')}
    r = s.post(f"{BASE_URL}/documents/upload", files=files)
    r.raise_for_status()
    doc_id = r.json().get('document_id')
    print(f" Uploaded successfully, ID: {doc_id}")

    print("4. Triggering Analysis...")
    r = s.post(f"{BASE_URL}/analysis/analyze/{doc_id}")
    r.raise_for_status()
    print(" Analysis triggered.")

    print("5. Waiting for analysis to finish...")
    for _ in range(30):
        r = s.get(f"{BASE_URL}/documents/{doc_id}")
        r.raise_for_status()
        status = r.json().get('status')
        if status == 'analyzed':
            print(" Analysis completed successfully!")
            break
        elif status == 'failed':
            print(" Analysis FAILED.")
            break
        time.sleep(2)
    else:
        print(" Analysis timed out.")

    print("6. Getting Analysis Results...")
    r = s.get(f"{BASE_URL}/analysis/{doc_id}")
    if r.status_code == 200:
        results = r.json()
        print(f" Risk level: {results.get('risk_level')}")
    else:
        print(f" Failed to get results: {r.text}")

    print("7. Testing Global Chat...")
    chat_payload = {"prompt": "What documents have high risk?"}
    r = s.post(f"{BASE_URL}/chat/query", json=chat_payload)
    if r.status_code == 200:
        print(" Chat Response:", r.json().get('response'))
    else:
        print(" Chat Failed:", r.text)

if __name__ == "__main__":
    run_test()
