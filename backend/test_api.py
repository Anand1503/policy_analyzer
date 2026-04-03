"""Quick API endpoint validation script."""
import requests

BASE = "http://localhost:8000/api/v1"

# 1. Login (JSON body, not form data)
print("=== 1. LOGIN ===")
r = requests.post(f"{BASE}/auth/login", json={"username": "admin", "password": "Admin1234"})
print(f"  Status: {r.status_code}")
if r.status_code != 200:
    print(f"  Response: {r.text[:200]}")
token = r.json().get("access_token", "")
print(f"  Token OK: {bool(token)}")
if not token:
    print("  LOGIN FAILED - cannot proceed")
    exit(1)
headers = {"Authorization": f"Bearer {token}"}

# 2. Get current user
print("\n=== 2. GET ME ===")
r = requests.get(f"{BASE}/auth/me", headers=headers)
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    me = r.json()
    print(f"  Username: {me.get('username')}")
    print(f"  Email: {me.get('email')}")

# 3. List documents
print("\n=== 3. LIST DOCUMENTS ===")
r = requests.get(f"{BASE}/documents/", headers=headers)
print(f"  Status: {r.status_code}")
docs = r.json().get("documents", [])
print(f"  Total: {len(docs)}")
for d in docs[:3]:
    fname = d.get("original_filename", "?")
    st = d.get("status", "?")
    ccount = d.get("total_clauses", 0)
    print(f"    {fname} -> {st} ({ccount} clauses)")

# 4. Analysis results for latest analyzed doc
analyzed = [d for d in docs if d["status"] == "analyzed"]
if analyzed:
    doc_id = analyzed[0]["id"]
    print(f"\n=== 4. ANALYSIS RESULTS ({doc_id[:8]}...) ===")
    r = requests.get(f"{BASE}/analysis/results/{doc_id}", headers=headers)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Risk: {data.get('overall_risk')}")
        print(f"  Score: {data.get('overall_score')}")
        print(f"  Clauses: {len(data.get('clauses', []))}")
        recs = data.get("recommendations", [])
        print(f"  Recommendations: {len(recs)}")
        summary = str(data.get("summary", ""))
        print(f"  Summary: {summary[:150]}")

    # 5. Chat (RAG)
    print(f"\n=== 5. CHAT (RAG) ===")
    r = requests.post(
        f"{BASE}/analysis/chat",
        json={"query": "What data does this policy collect?", "document_id": doc_id},
        headers=headers,
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        ans = r.json().get("answer", "")
        srcs = r.json().get("sources", [])
        print(f"  Answer: {ans[:200]}")
        print(f"  Sources: {len(srcs)}")
    else:
        print(f"  Error: {r.text[:200]}")

    # 6. Compliance
    print(f"\n=== 6. COMPLIANCE REPORT ===")
    r = requests.get(f"{BASE}/analysis/compliance/{doc_id}?framework=GDPR", headers=headers)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Score: {data.get('compliance_score')}")
        print(f"  Framework: {data.get('framework')}")
    else:
        print(f"  Error: {r.text[:200]}")

    # 7. Risk report
    print(f"\n=== 7. RISK REPORT ===")
    r = requests.get(f"{BASE}/analysis/report/{doc_id}", headers=headers)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  Keys: {list(r.json().keys())}")
    else:
        print(f"  Error: {r.text[:200]}")

    # 8. Entities
    print(f"\n=== 8. NER ENTITIES ===")
    r = requests.get(f"{BASE}/analysis/entities/{doc_id}", headers=headers)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        ents = data.get("entities", data)
        if isinstance(ents, dict):
            print(f"  Entity types: {len(ents)}")
            for etype, values in list(ents.items())[:5]:
                print(f"    {etype}: {values[:3] if isinstance(values, list) else values}")
        else:
            print(f"  Entities found: {len(ents)}")
            for e in list(ents)[:5]:
                print(f"    {e}")
    else:
        print(f"  Error: {r.text[:150]}")
else:
    print("\n  No analyzed documents found!")

# 9. Health
print("\n=== 9. HEALTH CHECK ===")
r = requests.get("http://localhost:8000/health/ready")
print(f"  Status: {r.status_code}")
checks = r.json().get("checks", {})
for k, v in checks.items():
    status_icon = "OK" if v in ("ok", "loaded", "available") else "WARN"
    print(f"    [{status_icon}] {k}: {v}")

print("\n========== ALL TESTS COMPLETE ==========")
