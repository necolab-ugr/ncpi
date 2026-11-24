"""Helpers to start WebUI jobs and wait for them through Flask's test client."""
import time


def start_job(client, computation_type, data, referer="/"):
    """POST a computation form and return the job id from the redirect."""
    resp = client.post(
        f"/start_computation_redirect/{computation_type}",
        data=data,
        headers={"Referer": referer},
        content_type="multipart/form-data",
    )
    location = resp.headers.get("Location", "")
    assert resp.status_code == 302 and "/job_status/" in location, (
        f"{computation_type} job was not started (status {resp.status_code}, location '{location}')."
    )
    return location.split("/job_status/", 1)[1].split("?", 1)[0]


def wait_job(client, job_id, timeout=180.0):
    """Poll /status/<job_id> until the job ends; fail unless it finished."""
    deadline = time.monotonic() + timeout
    while True:
        resp = client.get(f"/status/{job_id}")
        info = resp.get_json() or {}
        status = info.get("status")
        if status in {"finished", "failed", "cancelled"}:
            break
        assert time.monotonic() < deadline, f"Job {job_id} timed out (last status: {status})."
        time.sleep(0.25)
    assert status == "finished", (
        f"Job {job_id} ended with status '{status}': {info.get('error')}\n{info.get('output', '')}"
    )
    return info
