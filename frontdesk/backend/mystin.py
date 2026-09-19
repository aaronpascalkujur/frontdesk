"""Client for a locally running Mystin Office server."""

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:4521"


class MystinError(RuntimeError):
    pass


class MystinOffice:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def agents(self) -> list[dict]:
        try:
            r = requests.get(f"{self.base_url}/api/agents", timeout=10)
        except requests.RequestException as e:
            raise MystinError(f"cannot reach Mystin Office at {self.base_url}: {e}") from e
        if r.status_code != 200:
            raise MystinError(f"/api/agents returned {r.status_code}")
        return r.json()

    def run_task(self, agent_id: str, text: str) -> dict:
        """Dispatch a task. Blocks until the agent finishes; returns {agent, result, file}."""
        try:
            r = requests.post(
                f"{self.base_url}/api/task",
                json={"agentId": agent_id, "text": text},
                timeout=self.timeout,
            )
        except requests.Timeout as e:
            raise MystinError("the agent took too long to answer") from e
        except requests.RequestException as e:
            raise MystinError(f"cannot reach Mystin Office: {e}") from e

        if r.status_code != 200:
            try:
                raise MystinError(r.json().get("error", f"HTTP {r.status_code}"))
            except ValueError:
                raise MystinError(f"HTTP {r.status_code}") from None
        return r.json()
