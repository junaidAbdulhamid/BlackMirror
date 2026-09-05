from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from blackmirror.api.app import app, comparison_loader_dep
from blackmirror.api.loader import BinaryArray


class FakeComparisonLoader:
    def create(
        self, reference_run_id: str, candidate_run_ids: tuple[str, ...], **_: object
    ) -> dict[str, object]:
        return self.metadata(f"{reference_run_id}-vs-{candidate_run_ids[0]}")

    def list_comparisons(self) -> list[dict[str, object]]:
        return [self.metadata("cmp-1")]

    def metadata(self, comparison_id: str) -> dict[str, object]:
        if comparison_id == "missing":
            raise FileNotFoundError(comparison_id)
        return {
            "comparison_id": comparison_id,
            "metadata": {
                "comparison_version": "1.0",
                "reference_run_id": "run-a",
                "candidate_run_ids": ["run-b"],
            },
        }

    def array(self, comparison_id: str, candidate_run_id: str, key: str) -> BinaryArray:
        if candidate_run_id != "run-b" or key != "cortical_signed_delta":
            raise KeyError(key)
        values = np.array([[1.0, -2.0]], dtype="<f8")
        return BinaryArray(values.tobytes(), str(values.dtype), values.shape)


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[comparison_loader_dep] = FakeComparisonLoader
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_list_and_get_comparison(client: TestClient) -> None:
    assert client.get("/api/comparisons").json()[0]["comparison_id"] == "cmp-1"
    response = client.get("/api/comparisons/cmp-1")
    assert response.status_code == 200
    assert response.json()["metadata"]["reference_run_id"] == "run-a"


def test_create_comparison(client: TestClient) -> None:
    response = client.post(
        "/api/comparisons",
        json={"reference_run_id": "run-a", "candidate_run_ids": ["run-b"]},
    )
    assert response.status_code == 201
    assert response.json()["comparison_id"] == "run-a-vs-run-b"


@pytest.mark.parametrize(
    "payload",
    [
        {"reference_run_id": "..", "candidate_run_ids": ["run-b"]},
        {"reference_run_id": "run-a", "candidate_run_ids": ["../run-b"]},
        {
            "reference_run_id": "run-a",
            "candidate_run_ids": ["run-b"],
            "goal_score": "engagement",
        },
    ],
)
def test_create_comparison_rejects_unsafe_or_unsupported_fields(
    client: TestClient, payload: dict[str, object]
) -> None:
    assert client.post("/api/comparisons", json=payload).status_code == 422


def test_comparison_binary_array(client: TestClient) -> None:
    response = client.get("/api/comparisons/cmp-1/pairs/run-b/arrays/cortical_signed_delta")
    assert response.status_code == 200
    assert response.headers["X-Array-Shape"] == "1,2"
    np.testing.assert_array_equal(np.frombuffer(response.content, dtype="<f8"), [1.0, -2.0])


def test_missing_comparison_and_array_are_404(client: TestClient) -> None:
    assert client.get("/api/comparisons/missing").status_code == 404
    assert client.get("/api/comparisons/cmp-1/pairs/run-x/arrays/nope").status_code == 404
