import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

COMPLETED_STATE = "completed"
FAILED_STATES = {"compile_failed", "benchmark_failed"}
TERMINAL_STATES = FAILED_STATES | {COMPLETED_STATE}


class EvaluatorError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class EvaluatorClient:
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "EvaluatorClient":
        port = os.environ.get("KERNEL_EVALUATOR_PORT", "8000")
        base_url = os.environ.get("KERNEL_EVALUATOR_API", f"http://localhost:{port}")
        api_key = (
            os.environ.get("KERNEL_EVALUATOR_ADMIN_API_KEY")
            or os.environ.get("KERNEL_EVALUATOR_API_KEY")
        )
        if not api_key:
            raise KeyError("KERNEL_EVALUATOR_ADMIN_API_KEY or KERNEL_EVALUATOR_API_KEY must be set")
        return cls(base_url=base_url.rstrip("/"), api_key=api_key)

    def _raw(self, method: str, path: str, api_key: str | None = None,
             body: dict | None = None) -> bytes:
        headers = {"X-API-Key": api_key or self.api_key}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode() if exc.fp else ""
            raise EvaluatorError(f"{method} {path} -> {exc.code} {exc.reason}: {detail}", status=exc.code) from exc

    def get_json(self, path: str, api_key: str | None = None) -> dict:
        payload = self._raw("GET", path, api_key)
        return json.loads(payload) if payload else {}

    def get_text(self, path: str, api_key: str | None = None) -> str:
        return self._raw("GET", path, api_key).decode()

    def get_bytes(self, path: str, api_key: str | None = None) -> bytes:
        return self._raw("GET", path, api_key)

    def post_json(self, path: str, body: dict | None = None, api_key: str | None = None) -> dict:
        payload = self._raw("POST", path, api_key, body)
        return json.loads(payload) if payload else {}

    def patch_json(self, path: str, body: dict, api_key: str | None = None) -> dict:
        payload = self._raw("PATCH", path, api_key, body)
        return json.loads(payload) if payload else {}

    def create_run(self, plugin: str, target: str, shapes: list[dict], target_speedup: float | None = None,
                   benchmark_policy: dict | None = None) -> dict:
        body: dict = {"plugin": plugin, "target": target, "shapes": shapes}
        if target_speedup is not None:
            body["target_speedup"] = target_speedup
        if benchmark_policy is not None:
            body["benchmark_policy"] = benchmark_policy
        return self.post_json("/evaluation/runs", body)

    def get_run(self, run_id: str) -> dict:
        return self.get_json(f"/evaluation/runs/{run_id}")

    def mint_user_key(self) -> str:
        return self.post_json("/api-keys", {"role": "user"})["api_key"]

    def submit(self, run_id: str, source_text: str, api_key: str | None = None,
               artifacts: tuple[str, ...] = (), agent_index: int | None = None) -> dict:
        body: dict = {"source_text": source_text}
        if artifacts:
            body["artifacts"] = list(artifacts)
        if agent_index is not None:
            body["agent_index"] = agent_index
        return self.post_json(f"/evaluation/runs/{run_id}/jobs", body, api_key)

    def job(self, job_id: str, api_key: str | None = None) -> dict:
        return self.get_json(f"/evaluation/jobs/{job_id}", api_key)

    def job_result(self, job_id: str, api_key: str | None = None) -> dict:
        return self.get_json(f"/evaluation/jobs/{job_id}/result", api_key)

    def poll(self, job_id: str, timeout_s: float = 600.0, interval_s: float = 2.0,
             api_key: str | None = None) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            job = self.job(job_id, api_key)
            state = job["state"]
            if state == COMPLETED_STATE:
                return self.job_result(job_id, api_key)
            if state in FAILED_STATES:
                detail = job["compile_error"] or job["benchmark_error"] or state
                raise EvaluatorError(f"job {job_id} {state}: {detail}")
            time.sleep(interval_s)
        raise EvaluatorError(f"job {job_id} did not reach a terminal state within {timeout_s}s")

    def archive(self, run_id: str) -> list[dict]:
        return self.get_json(f"/scaffold/archive?run_id={run_id}")["entries"]

    def agent_bests(self, run_id: str) -> list[dict]:
        return self.get_json(f"/scaffold/agent-bests?run_id={run_id}")["agent_bests"]

    def best_kernel(self, run_id: str) -> dict | None:
        try:
            result = self.get_json(f"/evaluation/runs/{run_id}/best-kernel")
        except EvaluatorError:
            return None
        return result or None

    def patch_kernel(self, kernel_id: str, fields: dict) -> dict:
        return self.patch_json(f"/kernels/{kernel_id}", fields)
