"""Smoke test: every WebUI section page renders from a fresh session."""
import pytest

# (route, expected <title> prefix rendered by base.html)
PAGES = [
    ("/", "Dashboard"),
    ("/sessions", "Previous Sessions"),
    ("/simulation", "Simulation"),
    ("/simulation/upload_sim", "Simulation data upload"),
    ("/simulation/new_sim", "New simulation"),
    ("/simulation/new_sim/hagen", "New simulation"),
    ("/simulation/new_sim/cavallari", "New simulation"),
    ("/simulation/new_sim/four_area", "New simulation"),
    ("/simulation/new_sim/custom", "New simulation"),
    ("/field_potential", "Field Potential"),
    ("/field_potential/load", "Load Field Potential Data"),
    ("/field_potential/kernel", "Field Potential kernel"),
    ("/field_potential/proxy", "Field Potential proxy"),
    ("/features", "Features"),
    ("/features/methods", "Features"),
    ("/features/load_data", "Load Precomputed Features"),
    ("/inference", "Inference"),
    ("/inference/load_data", "Load Precomputed Predictions"),
    ("/inference/new_training", "New training"),
    ("/inference/compute_predictions", "Compute predictions"),
    ("/analysis", "Analysis"),
]


@pytest.mark.parametrize("route, title", PAGES, ids=[p[0] for p in PAGES])
def test_section_page_renders(client, route, title):
    resp = client.get(route)
    assert resp.status_code == 200, f"GET {route} returned {resp.status_code}"
    html = resp.get_data(as_text=True)
    assert f"<title>{title} - ncpi</title>" in html


@pytest.mark.parametrize(
    "method, route, expected_status",
    [
        ("get", "/status/unknown-job", 404),
        ("get", "/analysis/columns/current", 404),
        ("get", "/download_single_file", 400),
        ("post", "/start_computation_redirect/not_a_module", 400),
    ],
)
def test_stateless_endpoints_fail_cleanly(client, method, route, expected_status):
    resp = getattr(client, method)(route)
    assert resp.status_code == expected_status
