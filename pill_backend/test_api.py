#!/usr/bin/env python3
# test_api.py — Test the Pill Recognition API
# Run this AFTER the server is started with run.py

import requests
import sys
import json

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    r = requests.get(f"{API_URL}/health")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    print("  PASSED\n")


def test_stats():
    """Test stats endpoint."""
    print("Testing /stats...")
    r = requests.get(f"{API_URL}/stats")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    print("  PASSED\n")


def test_predict(image_path):
    """Test prediction with a real image."""
    print(f"Testing /predict with {image_path}...")

    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        r = requests.post(f"{API_URL}/predict", files=files)

    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Inference time: {data.get('inference_time_ms', '?')} ms")
    print(f"  Results: {data.get('num_results', 0)} matches\n")

    if data.get("results"):
        for res in data["results"]:
            print(f"  #{res['rank']} {res['drug_name'][:50]}")
            print(f"     Similarity: {res['similarity']}%")
            print(f"     Shape: {res['shape']}, Colors: {res['colors']}")
            print(f"     Imprint: {res['imprint']}")
            print()

    assert r.status_code == 200
    print("  PASSED\n")


def test_invalid_file():
    """Test with invalid file type."""
    print("Testing /predict with invalid file...")
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    r = requests.post(f"{API_URL}/predict", files=files)
    print(f"  Status: {r.status_code} (expected 400)")
    assert r.status_code == 400
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 50)
    print("PILL RECOGNITION API — TEST SUITE")
    print("=" * 50 + "\n")

    test_health()
    test_stats()
    test_invalid_file()

    # Test with real image if provided
    if len(sys.argv) > 1:
        test_predict(sys.argv[1])
    else:
        print("To test prediction, run:")
        print("  python test_api.py path/to/pill_image.jpg")

    print("All tests passed!")
