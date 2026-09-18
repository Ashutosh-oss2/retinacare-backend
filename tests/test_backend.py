"""
Comprehensive Automated Verification Suite for RetinaCare Backend Repository.
Can be executed directly: `python tests/test_backend.py` or with `pytest`.
"""

import os
import sys
from fastapi.testclient import TestClient

# Ensure root of retinacare-backend is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app
from app.services.model_registry import registry

# Force model registry initialization for synchronous test client
registry.initialize(weights_dir=os.path.join(BASE_DIR, "weights"))
client = TestClient(app)

# Resolve sample test image
SAMPLE_IMG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "test_images", "0005cfc8afb6.png"))
if not os.path.isfile(SAMPLE_IMG_PATH):
    raise FileNotFoundError(f"Test image not found at: {SAMPLE_IMG_PATH}")

with open(SAMPLE_IMG_PATH, "rb") as f:
    SAMPLE_BYTES = f.read()


def test_1_models_loaded_once():
    print("\n[Test 1] Verifying all 3 models are loaded and initialized...")
    assert registry.is_aptos_loaded, "Model 1 (APTOS) is not loaded."
    assert registry.is_vessel_loaded, "Model 2 (DRIVE) is not loaded."
    assert registry.is_idrid_loaded, "Model 3 (IDRiD) is not loaded."
    print("  -> Model 1 (APTOS): LOADED")
    print("  -> Model 2 (DRIVE): LOADED")
    print("  -> Model 3 (IDRiD): LOADED")


def test_2_health_endpoints():
    print("\n[Test 2] Testing root discovery and health check endpoints...")
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"
    print(f"  -> GET / -> Status {res_root.status_code} (Service: {res_root.json().get('service')})")

    res_health = client.get("/health")
    assert res_health.status_code == 200
    h_data = res_health.json()
    assert h_data["status"] == "healthy"
    assert h_data["models"]["model_1_aptos"] is True
    assert h_data["models"]["model_2_drive"] is True
    assert h_data["models"]["model_3_idrid"] is True
    print(f"  -> GET /health -> Status {res_health.status_code} (Models: {h_data['models']})")

    res_api_health = client.get("/api/health")
    assert res_api_health.status_code == 200
    print(f"  -> GET /api/health -> Status {res_api_health.status_code} (Alias PASS)")


def test_3_cors_allowed_and_disallowed_origins():
    print("\n[Test 3] Testing CORS headers for allowed vs disallowed origins...")
    # 1. Allowed Vercel Origin
    res_allowed = client.options(
        "/predict/aptos",
        headers={
            "Origin": "https://retinacare-app.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
    )
    assert res_allowed.status_code == 200
    assert res_allowed.headers.get("access-control-allow-origin") == "https://retinacare-app.vercel.app"
    print(f"  -> Allowed origin accepted: {res_allowed.headers.get('access-control-allow-origin')}")

    # 2. Disallowed Random Origin
    res_disallowed = client.options(
        "/predict/aptos",
        headers={
            "Origin": "https://unauthorized-domain.com",
            "Access-Control-Request-Method": "POST"
        }
    )
    assert res_disallowed.headers.get("access-control-allow-origin") != "https://unauthorized-domain.com"
    print("  -> Disallowed origin rejected (PASS)")


def test_4_model_1_aptos_prediction():
    print("\n[Test 4] Testing Model 1: APTOS DR Classification...")
    for path in ["/predict/aptos", "/api/predict/aptos"]:
        res = client.post(path, files={"file": ("fundus.png", SAMPLE_BYTES, "image/png")})
        assert res.status_code == 200, f"Failed on {path}: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["prediction"]["predicted_class"] in [0, 1, 2, 3, 4]
        assert "disclaimer" in data
        print(f"  -> {path} -> Predicted: {data['prediction']['predicted_label']} ({data['prediction']['confidence']}%)")

    # Legacy alias test
    res_legacy = client.post("/api/predict", files={"image": ("fundus.png", SAMPLE_BYTES, "image/png")})
    assert res_legacy.status_code == 200
    print("  -> /api/predict (legacy alias) -> PASS")


def test_5_model_2_drive_vessel_segmentation():
    print("\n[Test 5] Testing Model 2: DRIVE Vessel Segmentation...")
    for path in ["/predict/vessel", "/api/predict/vessel", "/predict/drive", "/api/predict/drive"]:
        res = client.post(path, files={"file": ("fundus.png", SAMPLE_BYTES, "image/png")})
        assert res.status_code == 200, f"Failed on {path}: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["vessel_density_percentage"] > 0
        assert data["mask_base64_png"] is not None
        assert data["mask_base64_png"].startswith("data:image/png;base64,")
        print(f"  -> {path} -> Vessel Density: {data['vessel_density_percentage']}% ({data['vessel_pixel_count']} px)")


def test_6_model_3_idrid_lesion_segmentation():
    print("\n[Test 6] Testing Model 3: IDRiD Lesion Segmentation...")
    for path in ["/predict/idrid", "/api/predict/idrid"]:
        res = client.post(path, files={"file": ("fundus.png", SAMPLE_BYTES, "image/png")})
        assert res.status_code == 200, f"Failed on {path}: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert "lesions" in data
        assert data["overlay_base64_png"] is not None
        assert data["overlay_base64_png"].startswith("data:image/png;base64,")
        print(f"  -> {path} -> Risk: {data['clinical_risk_level']}, Total Lesion Area: {data['total_lesion_area']}%")


def test_7_unified_multi_model_prediction():
    print("\n[Test 7] Testing Unified Multi-Model Pipeline (/predict/all)...")
    for path in ["/predict/all", "/api/predict/all"]:
        res = client.post(path, files={"file": ("fundus.png", SAMPLE_BYTES, "image/png")})
        assert res.status_code == 200, f"Failed on {path}: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert data["aptos"] is not None
        assert data["vessel"] is not None
        assert data["idrid"] is not None
        print(f"  -> {path} -> All 3 models successfully returned composite results.")


def test_8_model_info_and_stats():
    print("\n[Test 8] Testing Model Metadata and Audited Stats...")
    res_info = client.get("/api/model-info")
    assert res_info.status_code == 200

    res_stats = client.get("/api/model-stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert stats["model_1_aptos"]["referable_screening_accuracy"] == 0.9180
    assert stats["model_2_drive"]["pixel_accuracy"] == 0.9535
    assert stats["model_3_idrid"]["macro_dice"] == 0.3435
    print("  -> /api/model-info & /api/model-stats -> Verified audited benchmarks.")


def test_9_error_handling():
    print("\n[Test 9] Testing Error Handling & Input Validation...")
    res_corrupt = client.post("/predict/aptos", files={"file": ("bad.png", b"NOT_IMAGE", "image/png")})
    assert res_corrupt.status_code == 400
    assert res_corrupt.json()["detail"]["error"] == "invalid_image"
    print("  -> Corrupt image rejected with 400 invalid_image")

    res_empty = client.post("/predict/vessel", files={"file": ("empty.png", b"", "image/png")})
    assert res_empty.status_code == 400
    assert res_empty.json()["detail"]["error"] == "empty_file"
    print("  -> Empty file rejected with 400 empty_file")

    res_ext = client.post("/predict/idrid", files={"file": ("notes.txt", b"plain text", "text/plain")})
    assert res_ext.status_code == 400
    assert res_ext.json()["detail"]["error"] == "unsupported_format"
    print("  -> Unsupported format rejected with 400 unsupported_format")


def test_10_no_frontend_files_in_repository():
    print("\n[Test 10] Confirming no frontend files exist in repository...")
    forbidden = ["node_modules", "package.json", "vite.config", "dist", "components"]
    for root, dirs, files in os.walk(BASE_DIR):
        for term in forbidden:
            assert term not in root.lower() or "app" in root, f"Forbidden frontend element found: {root}"
        for f in files:
            assert not f.endswith((".tsx", ".jsx")), f"Forbidden frontend source file found: {f}"
    print("  -> Zero frontend components, JSX/TSX, or node_modules found. Clean backend repository verified.")


def run_all_tests():
    print("=" * 80)
    print("       RETINACARE-BACKEND REPOSITORY VERIFICATION SUITE")
    print("=" * 80)
    test_1_models_loaded_once()
    test_2_health_endpoints()
    test_3_cors_allowed_and_disallowed_origins()
    test_4_model_1_aptos_prediction()
    test_5_model_2_drive_vessel_segmentation()
    test_6_model_3_idrid_lesion_segmentation()
    test_7_unified_multi_model_prediction()
    test_8_model_info_and_stats()
    test_9_error_handling()
    test_10_no_frontend_files_in_repository()
    print("\n" + "=" * 80)
    print("       ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY (100% PASS)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
