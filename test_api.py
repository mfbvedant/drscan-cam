"""DRScan Cam — End-to-end test script"""
import requests
import json

BASE = "http://127.0.0.1:8080"
IMG = r"c:\Users\VEDAN\drscan cam\frontend\assets\sample_xray.png"

# Test 1: Health
h = requests.get(f"{BASE}/api/health").json()
print("=== Health Check ===")
print(f"  Status: {h['status']}")
print(f"  Mode: {h['inference_mode']}")
print(f"  Modalities: {h['modalities']}")

# Test 2: Chest X-Ray
files = {"file": ("sample.png", open(IMG, "rb"), "image/png")}
r = requests.post(f"{BASE}/api/predict", files=files, data={"modality": "chest_xray"}).json()
print(f"\n=== Chest X-Ray ===")
print(f"  Diagnosis: {r['diagnosis']} ({r['confidence']*100:.1f}%)")
print(f"  Severity: {r['severity_label']} (Grade {r['severity']})")
print(f"  Model: {r['model_version']}")
print(f"  Time: {r['inference_time_ms']:.0f}ms")
print(f"  Heatmap: {'YES' if r.get('heatmap_base64') else 'NO'}")
for p in r["top_predictions"][:3]:
    print(f"    {p['class_name']}: {p['confidence']*100:.1f}%")

# Test 3: Brain MRI
files = {"file": ("sample.png", open(IMG, "rb"), "image/png")}
r2 = requests.post(f"{BASE}/api/predict", files=files, data={"modality": "brain_mri"}).json()
print(f"\n=== Brain MRI ===")
print(f"  Diagnosis: {r2['diagnosis']} ({r2['confidence']*100:.1f}%)")

# Test 4: Lung CT
files = {"file": ("sample.png", open(IMG, "rb"), "image/png")}
r3 = requests.post(f"{BASE}/api/predict", files=files, data={"modality": "lung_ct"}).json()
print(f"\n=== Lung CT ===")
print(f"  Diagnosis: {r3['diagnosis']} ({r3['confidence']*100:.1f}%)")

print("\n=== ALL TESTS PASSED ===")
