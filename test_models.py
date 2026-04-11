"""
DRScan AI — Full Model Verification Script
Tests all 4 modalities to confirm every model is connected and responding.
"""
import requests
import sys

API = "http://127.0.0.1:8080"
IMG = "frontend/assets/sample_xray.png"

def test_health():
    print("=== HEALTH CHECK ===")
    r = requests.get(f"{API}/api/health")
    h = r.json()
    print(f"  Version: {h['version']}")
    print(f"  Inference mode: {h['inference_mode']}")
    print(f"  Modalities: {h['modalities']}")
    print()
    return h

def test_modality(mod):
    print(f"=== {mod.upper()} ===")
    try:
        with open(IMG, "rb") as f:
            files = {"file": ("sample.png", f, "image/png")}
            r = requests.post(f"{API}/api/predict", files=files, data={"modality": mod}, timeout=120)

        if r.status_code == 200:
            d = r.json()
            print(f"  Status: OK")
            print(f"  Model: {d['model_version']}")
            print(f"  Diagnosis: {d['diagnosis']}")
            print(f"  Severity: {d['severity']} ({d['severity_label']})")
            print(f"  Confidence: {d['confidence']}")
            has_heatmap = "YES" if d.get("heatmap_base64") else "NO"
            print(f"  Heatmap: {has_heatmap}")
            regions = d.get("regions_of_concern", [])
            print(f"  Regions: {len(regions)} found")
            urgency = d.get("urgency", "N/A")
            print(f"  Urgency: {urgency}")
            print(f"  Time: {d['inference_time_ms']}ms")
            return True
        else:
            print(f"  ERROR {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

if __name__ == "__main__":
    health = test_health()

    modalities = ["chest_xray", "brain_mri", "lung_ct", "retinal_fundus"]
    results = {}

    for mod in modalities:
        results[mod] = test_modality(mod)
        print()

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for mod, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {mod:20s} -> {status}")

    all_pass = all(results.values())
    print()
    print(f"Result: {'ALL MODELS CONNECTED' if all_pass else 'SOME MODELS FAILED'}")
    sys.exit(0 if all_pass else 1)
