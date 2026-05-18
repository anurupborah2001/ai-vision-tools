import base64

import cv2
from fastapi.testclient import TestClient

from visionflow.api import create_app


def _encode_frame(frame):
    ok, encoded = cv2.imencode(".png", frame)
    assert ok is True
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def test_health_endpoint():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_catalog_lists_component_categories():
    client = TestClient(create_app())

    response = client.get("/api/v1/catalog")

    assert response.status_code == 200
    body = response.json()
    assert "categories" in body
    assert "preprocessing" in body["categories"]
    assert "augmentations" in body["categories"]
    assert "components" in body["categories"]


def test_preprocessing_endpoint_returns_serialized_image(sample_frame):
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/preprocessing/AutoOrient",
        json={
            "image_base64": _encode_frame(sample_frame),
            "init_args": {"rotation": 90},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "AutoOrient"
    assert body["result"]["frame"]["type"] == "image"
    assert body["result"]["frame"]["shape"] == [60, 40, 3]


def test_augmentation_endpoint_supports_payload_images(sample_frame):
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/augmentations/Cutout",
        json={
            "payload": {"frame_base64": _encode_frame(sample_frame)},
            "config": {
                "cutout_x": 5,
                "cutout_y": 5,
                "cutout_width": 10,
                "cutout_height": 10,
                "cutout_fill_value": [0, 0, 0],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Cutout"
    assert body["result"]["frame"]["type"] == "image"
    assert body["result"]["frame"]["shape"] == [40, 60, 3]


def test_component_endpoint_supports_payload_style_execution(sample_frame):
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/components/FrameEnhancer",
        json={
            "payload": {"frame_base64": _encode_frame(sample_frame)},
            "config": {"brightness": 10, "contrast": 1.1},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "FrameEnhancer"
    assert body["result"]["frame"]["type"] == "image"
    assert body["result"]["frame"]["shape"] == [40, 60, 3]


def test_unknown_component_returns_404():
    client = TestClient(create_app())

    response = client.post("/api/v1/preprocessing/DoesNotExist", json={})

    assert response.status_code == 404
