from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from blackmirror.api.app import analytics_loader_dep, app
from blackmirror.api.loader import BinaryArray


class FakeAnalyticsLoader:
    def metadata(self, run_id: str) -> dict[str, object]:
        if run_id == "missing":
            raise FileNotFoundError(run_id)
        return {
            "run_id": run_id,
            "metadata": {"analytics_version": "1.0"},
            "interpretation_notice": "mathematical summaries only",
        }

    def array(self, run_id: str, key: str) -> BinaryArray:
        if key != "global_mean":
            raise KeyError(key)
        values = np.array([1.0, 2.0], dtype="<f8")
        return BinaryArray(values.tobytes(), str(values.dtype), values.shape)


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[analytics_loader_dep] = FakeAnalyticsLoader
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_analytics_metadata_endpoint(client: TestClient) -> None:
    response = client.get("/api/runs/run-1/analytics")
    assert response.status_code == 200
    assert response.json()["metadata"]["analytics_version"] == "1.0"


def test_analytics_array_binary_transport(client: TestClient) -> None:
    response = client.get("/api/runs/run-1/analytics/arrays/global_mean")
    assert response.status_code == 200
    assert response.headers["X-Array-Dtype"] == "float64"
    assert response.headers["X-Array-Shape"] == "2"
    np.testing.assert_array_equal(np.frombuffer(response.content, dtype="<f8"), [1.0, 2.0])


def test_unknown_analytics_array_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/run-1/analytics/arrays/nope").status_code == 404


def test_missing_analytics_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/missing/analytics").status_code == 404
