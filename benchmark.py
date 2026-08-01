from pathlib import Path
import argparse
import json
import urllib.request
import urllib.parse

from llmark.llmark import LLMark

class RootcomputerBackend:
    """POST to a rootcomputer endpoint.

    Body:    {history, temperature, top_p, top_k, max_new_tokens, ...}
    Reply:   {reply: "..."}
    """

    def __init__(self, url: str, model: str, timeout: float = 180):
        self.url = url
        self.timeout = timeout
        self.model = model

    def complete(self, prompt: str, max_new_tokens: int = 20) -> str:
        payload = {
            "history": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.0,
            "no_repeat_ngram": 0,
        }
        reply = self._post(payload)

        return reply or ""

    def _post(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": (
                "Rootcomputerdev benchmark client"
            ),
        }

        headers["Origin"] = self.url
        headers["Referer"] = self.url
        headers["X-Requested-With"] = "XMLHttpRequest"

        endpoint = self.url + "/api/" + self.model + ".php"
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict):
                reply = data.get("reply", "")
                return reply if isinstance(reply, str) else None
        except Exception:
            return None
        return None

    def cleanup(self):
        pass


def make_backend(cfg: dict):
    """Construct a backend from a model config dict."""
    backend_type = cfg.get("backend")
    if backend_type == "root":
        backend = RootcomputerBackend(
            url=cfg["url"],
            model=cfg["model"],
            timeout=cfg.get("timeout", 180),
        )
        return backend

    raise ValueError(f"unknown backend type: {backend_type!r}")

def main():
    parser = argparse.ArgumentParser(
        description="Rootcomputerdev benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", default="benchmark.jsonl",
                        help="Test JSONL file (default: benchmark.json)")
    parser.add_argument("--model", default=None,
                        help="Model to run the tests on")
    parser.add_argument("--backend", default=None,
                        help="Backend (valid backends: [ rootcomputer ])")
    parser.add_argument("--url", default=None,
                        help="URL of the API")
    parser.add_argument("--out", default="results.json",
                        help="Output file (default: results.json)")
    args = parser.parse_args()

    cfg = {"test": args.test, "model": args.model, "backend": args.backend, "url": args.url, "out": args.out}

    backend = make_backend(cfg)

    test_path = Path(cfg["test"])
    base_path = test_path.parent.resolve()

    def chat(input: str) -> str:
        return backend.complete(input)

    llmark = LLMark(chat, str(base_path))

    with test_path.open(encoding="utf-8") as f:
        data: dict[str, any] = json.load(f)

    results = llmark.run_test_set(data)

    with open(cfg["out"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()