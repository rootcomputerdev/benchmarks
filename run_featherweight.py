"""
run_featherweight.py — Standalone runner for the Featherweight v1 benchmark.

Runs `featherweight.jsonl` (or any compatible JSONL test file) against any
number of small language models, sequentially, and produces:

  - One JSON file per model with every question's prediction and raw output
  - A summary CSV with one row per model and per-category accuracy columns
  - A summary JSON for programmatic consumption
  - A pretty-printed comparison table on stdout, sorted by parameter count

This tool has no dependency on ChatHaiku CLI. It reuses only the prompt
format and extraction logic from the evaluator plugin so that h2 results
from this runner are directly comparable to `/eval run featherweight`
inside the ChatHaiku dev client.

Usage:
  python run_featherweight.py
  python run_featherweight.py --test featherweight.jsonl --models models.json
  python run_featherweight.py --only h2,gpt2 --limit 20
  python run_featherweight.py --prompt-style completion

Backends:
  - http         POST {history, temperature, max_new_tokens, ...} -> {reply}
                 (matches the chathaiku endpoint)
  - huggingface  Local model via transformers
  - openai       OpenAI-compatible chat-completions (vLLM, ollama, LM Studio)

See models.example.json for a curated comparison set of featherweight-class
models (GPT-2 family, Pythia, OPT, SmolLM, Qwen2.5-0.5B, TinyLlama).
"""

import os
import sys
import re
import csv
import gc
import json
import time
import argparse
import urllib.request
import urllib.parse
from datetime import datetime, timezone


# ──────────────────────────────────────────────────────────
#  Prompt templates — `plain` matches the ChatHaiku evaluator
#  plugin exactly, so HTTP-backend results line up with the
#  in-chat /eval run featherweight numbers.
# ──────────────────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "plain": (
        "{question}\n\n"
        "{choices_block}\n\n"
        "Answer with just the letter of the correct choice."
    ),
    "completion": (
        "Question: {question}\n"
        "{choices_block}\n"
        "Answer:"
    ),
}


def format_prompt(question: str, choices: list, style: str = "plain") -> str:
    choices_block = "\n".join(
        f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)
    )
    template = PROMPT_TEMPLATES.get(style, PROMPT_TEMPLATES["plain"])
    return template.format(question=question, choices_block=choices_block)


# ──────────────────────────────────────────────────────────
#  Answer parsing — ported verbatim from the evaluator plugin
#  so behavior matches the in-chat runs.
# ──────────────────────────────────────────────────────────

def normalize_answer(ans, num_choices: int) -> str:
    max_letter = chr(ord('A') + num_choices - 1)
    if isinstance(ans, bool):
        raise ValueError(f"answer cannot be bool ({ans!r})")
    if isinstance(ans, int):
        if 0 <= ans < num_choices:
            return chr(ord('A') + ans)
        raise ValueError(f"answer index {ans} out of range 0..{num_choices-1}")
    if isinstance(ans, str):
        s = ans.strip().upper()
        if len(s) == 1 and 'A' <= s <= max_letter:
            return s
        m = re.fullmatch(r'\(?([A-Z])\)?\.?', s)
        if m and 'A' <= m.group(1) <= max_letter:
            return m.group(1)
    raise ValueError(f"invalid answer {ans!r} for {num_choices} choices")


def _find_choice_position(text: str, choice_text: str):
    if not choice_text or not text:
        return None
    choice_text = choice_text.strip()
    if not choice_text:
        return None
    pattern = re.escape(choice_text)
    if choice_text[0].isalnum():
        pattern = r'\b' + pattern
    if choice_text[-1].isalnum():
        pattern = pattern + r'\b'
    m = re.search(pattern, text, re.IGNORECASE)
    return m.start() if m else None


def extract_letter(text: str, num_choices: int, choices=None):
    """Pull a letter (A..max) from a model reply. None if extraction fails.

    Strategy (in order):
      1. Parenthesized: "(A)"
      2. "answer: A" or "answer is A"
      3. Reply starts with "A)", "A.", "A,"
      4. Choice-text match (e.g. "Paris" matches choice B if choices[1]=="Paris")
      5. Any standalone letter A..max
      6. First non-whitespace character
    """
    if not text:
        return None
    max_letter = chr(ord('A') + num_choices - 1)
    pat = f'[A-{max_letter}]'
    flags = re.IGNORECASE

    m = re.search(r'\((' + pat + r')\)', text, flags)
    if m:
        return m.group(1).upper()
    m = re.search(r'answer\s*(?:is)?\s*[:\-]?\s*(' + pat + r')\b', text, flags)
    if m:
        return m.group(1).upper()
    m = re.match(r'^\s*(' + pat + r')[\s.,):]', text, flags)
    if m:
        return m.group(1).upper()

    if choices:
        matches = []
        for i, choice in enumerate(choices):
            pos = _find_choice_position(text, str(choice))
            if pos is not None:
                matches.append((pos, chr(ord('A') + i)))
        if matches:
            matches.sort()
            return matches[0][1]

    m = re.search(r'\b(' + pat + r')\b', text, flags)
    if m:
        return m.group(1).upper()
    stripped = text.strip()
    if stripped:
        first = stripped[0].upper()
        if 'A' <= first <= max_letter:
            return first
    return None


# ──────────────────────────────────────────────────────────
#  Test loader
# ──────────────────────────────────────────────────────────

def load_test(path: str):
    """Returns (meta_dict, items_list). Raises on invalid input."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    meta = {}
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {line_no} JSON: {e}")
            if obj.get("_meta"):
                meta = obj
                continue
            # Validate
            q = obj.get("question")
            choices = obj.get("choices")
            ans = obj.get("answer")
            if not isinstance(q, str) or not q.strip():
                raise ValueError(f"line {line_no}: missing/invalid 'question'")
            if not isinstance(choices, list) or len(choices) < 2:
                raise ValueError(f"line {line_no}: 'choices' must be a list of 2+")
            if ans is None:
                raise ValueError(f"line {line_no}: missing 'answer'")
            normalize_answer(ans, len(choices))  # raises on bad answer
            items.append(obj)
    return meta, items


# ──────────────────────────────────────────────────────────
#  Backends
# ──────────────────────────────────────────────────────────

class HttpBackend:
    """POST to a chathaiku-compatible endpoint.

    Body:    {history, temperature, top_p, top_k, max_new_tokens, ...}
    Reply:   {reply: "..."}
    """

    def __init__(self, url: str, fallback_url: str = None, timeout: float = 180):
        self.url = url
        self.fallback_url = fallback_url
        self.timeout = timeout
        self.params_count = None

    def info(self) -> dict:
        return {"backend": "http", "url": self.url}

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
        reply = self._post(self.url, payload)
        if reply is None and self.fallback_url:
            reply = self._post(self.fallback_url, payload)
        return reply or ""

    def _post(self, url: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        }
        # chathaiku.com requires browser-shaped headers to satisfy ModSecurity
        try:
            host = urllib.parse.urlsplit(url).netloc.lower()
        except Exception:
            host = ""
        if host.endswith("chathaiku.com"):
            headers["Origin"] = "https://chathaiku.com"
            headers["Referer"] = "https://chathaiku.com/"
            headers["X-Requested-With"] = "XMLHttpRequest"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
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


class HuggingFaceBackend:
    """Run a HuggingFace causal LM locally via transformers."""

    def __init__(self, model_id: str, dtype: str = "auto", device: str = "auto",
                 chat_template: bool = False, prompt_suffix: str = ""):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch required for huggingface backend. "
                "Install: pip install transformers torch"
            ) from e

        self._torch = torch
        self.model_id = model_id
        self.chat_template = chat_template
        self.prompt_suffix = prompt_suffix

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype == "auto":
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            torch_dtype = dtype_map.get(dtype, torch.float32)

        print(f"  Loading {model_id} on {self.device} ({torch_dtype})...",
              end="", flush=True)
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.params_count = sum(p.numel() for p in self.model.parameters())
        print(f" loaded {self.params_count:,} params in {time.time() - t0:.1f}s")

    def info(self) -> dict:
        return {
            "backend": "huggingface",
            "model_id": self.model_id,
            "device": self.device,
            "params": self.params_count,
            "chat_template": self.chat_template,
        }

    def complete(self, prompt: str, max_new_tokens: int = 20) -> str:
        torch = self._torch
        full_prompt = prompt + self.prompt_suffix

        # Tokenize. Apply chat template if requested AND available.
        used_template = False
        if self.chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": full_prompt}]
                input_ids = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True,
                ).to(self.device)
                used_template = True
            except Exception:
                used_template = False

        if not used_template:
            input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def cleanup(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class OpenAIBackend:
    """OpenAI-compatible /v1/chat/completions (works with vLLM, ollama, LM Studio)."""

    def __init__(self, url: str, model_id: str, api_key: str = None, timeout: float = 180):
        self.url = url
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self.params_count = None

    def info(self) -> dict:
        return {"backend": "openai", "url": self.url, "model": self.model_id}

    def complete(self, prompt: str, max_new_tokens: int = 20) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

    def cleanup(self):
        pass


def make_backend(cfg: dict):
    """Construct a backend from a model config dict."""
    backend_type = cfg.get("backend", "huggingface")
    if backend_type == "http":
        backend = HttpBackend(
            url=cfg["url"],
            fallback_url=cfg.get("fallback_url"),
            timeout=cfg.get("timeout", 180),
        )
        if cfg.get("params"):
            backend.params_count = cfg["params"]
        return backend
    elif backend_type == "huggingface":
        return HuggingFaceBackend(
            model_id=cfg["model_id"],
            dtype=cfg.get("dtype", "auto"),
            device=cfg.get("device", "auto"),
            chat_template=cfg.get("chat_template", False),
            prompt_suffix=cfg.get("prompt_suffix", ""),
        )
    elif backend_type == "openai":
        backend = OpenAIBackend(
            url=cfg["url"],
            model_id=cfg.get("model_id", cfg["name"]),
            api_key=cfg.get("api_key"),
            timeout=cfg.get("timeout", 180),
        )
        if cfg.get("params"):
            backend.params_count = cfg["params"]
        return backend
    raise ValueError(f"unknown backend type: {backend_type!r}")


# ──────────────────────────────────────────────────────────
#  Single-model run
# ──────────────────────────────────────────────────────────

def run_one_model(name: str, backend, questions: list, prompt_style: str = "plain"):
    total = len(questions)
    correct = 0
    parse_errors = 0
    by_cat = {}
    details = []
    t_start = time.time()

    for i, q in enumerate(questions, 1):
        num_choices = len(q["choices"])
        expected = normalize_answer(q["answer"], num_choices)
        prompt = format_prompt(q["question"], q["choices"], style=prompt_style)

        try:
            reply = backend.complete(prompt, max_new_tokens=20) or ""
        except KeyboardInterrupt:
            print()  # commit progress line before re-raising
            raise
        except Exception as e:
            reply = f"[backend error: {e}]"

        predicted = extract_letter(reply, num_choices, q["choices"])
        is_correct = (predicted is not None and predicted == expected)

        if is_correct:
            correct += 1
        if predicted is None:
            parse_errors += 1

        cat = q.get("category", "(none)")
        slot = by_cat.setdefault(cat, {"total": 0, "correct": 0})
        slot["total"] += 1
        if is_correct:
            slot["correct"] += 1

        details.append({
            "id": q.get("id", f"q{i}"),
            "category": q.get("category"),
            "question": q["question"],
            "choices": q["choices"],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "raw_response": reply,
        })

        # Rolling single-line progress (\r overwrite, \x1b[K clear-to-EOL)
        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed > 0 else 0
        eta_sec = (total - i) / rate if rate > 0 else 0
        eta_str = f"{int(eta_sec // 60)}:{int(eta_sec % 60):02d}"
        acc_so_far = correct / i
        line = (f"  {i}/{total}  "
                f"{correct} correct ({acc_so_far * 100:.1f}%)  "
                f"{rate:.1f} q/s  ETA {eta_str}")
        print("\r" + line + "\x1b[K", end="", flush=True)

    print()  # commit final progress line
    elapsed = time.time() - t_start

    for slot in by_cat.values():
        slot["accuracy"] = slot["correct"] / slot["total"] if slot["total"] > 0 else 0

    accuracy = correct / total if total > 0 else 0
    answered = total - parse_errors
    accuracy_parsed = correct / answered if answered > 0 else 0

    return {
        "model": name,
        "backend_info": backend.info(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompt_style": prompt_style,
        "total": total,
        "answered": answered,
        "correct": correct,
        "parse_errors": parse_errors,
        "accuracy": accuracy,
        "accuracy_parsed_only": accuracy_parsed,
        "elapsed_seconds": round(elapsed, 2),
        "by_category": by_cat,
        "details": details,
    }


# ──────────────────────────────────────────────────────────
#  Summary output
# ──────────────────────────────────────────────────────────

def format_params(n):
    if n is None:
        return "?"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    return str(n)


CATEGORY_ABBREV = {
    "geography": "Geo", "history": "Hst", "arithmetic": "Ari",
    "science": "Sci", "language": "Lng", "vocabulary": "Voc",
    "common-sense": "CS", "logic": "Log", "categorization": "Cat",
    "reading": "Rd",
}


def write_summary_csv(summaries: list, path: str, categories: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model", "params", "total", "correct", "accuracy",
                  "parse_errors", "elapsed_seconds"] + list(categories)
        writer.writerow(header)
        for s in summaries:
            row = [
                s["model"],
                s.get("params") or "",
                s["total"],
                s["correct"],
                f"{s['accuracy']:.4f}",
                s.get("parse_errors", 0),
                s.get("elapsed_seconds", 0),
            ]
            for cat in categories:
                slot = (s.get("by_category") or {}).get(cat, {})
                acc = slot.get("accuracy", 0) if slot else 0
                row.append(f"{acc:.4f}")
            writer.writerow(row)


def print_summary_table(summaries: list, categories: list, prompt_style: str,
                         test_name: str):
    print()
    print("=" * 90)
    print(" Featherweight v1 — Results")
    print("=" * 90)
    print(f"  Test:         {test_name}")
    print(f"  Prompt:       {prompt_style}")
    print(f"  Models run:   {len(summaries)}")
    print(f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Partition into ran-successfully and errored
    ok = [s for s in summaries if "error" not in s]
    bad = [s for s in summaries if "error" in s]

    # Sort by parameter count ascending; unknowns last
    def sort_key(s):
        p = s.get("params")
        return (p is None, p or 0)

    ok = sorted(ok, key=sort_key)

    if ok:
        name_w = max(18, max((len(s["model"]) for s in ok), default=18))
        cat_cols = "  ".join(
            f"{CATEGORY_ABBREV.get(c, c[:3]):>3}" for c in categories
        )

        print(f"  {'Model':<{name_w}}  {'Params':>6}  {'Score':>9}  {'Acc':>6}   {cat_cols}")
        print(f"  {'-' * name_w}  {'-' * 6}  {'-' * 9}  {'-' * 6}   "
              f"{'-' * len(cat_cols)}")

        for s in ok:
            params = format_params(s.get("params"))
            score = f"{s['correct']:>3}/{s['total']:<3}"
            acc = f"{s['accuracy'] * 100:5.1f}%"
            cats_str = "  ".join(
                f"{int((s.get('by_category') or {}).get(c, {}).get('accuracy', 0) * 100):>2}%"
                for c in categories
            )
            print(f"  {s['model']:<{name_w}}  {params:>6}  {score:>9}  "
                  f"{acc:>6}   {cats_str}")

        print()
        best = max(ok, key=lambda s: s["accuracy"])
        print(f"  Best: {best['model']} at {best['accuracy'] * 100:.1f}%")

    if bad:
        print()
        print("  Models that failed:")
        for s in bad:
            print(f"    {s['model']}: {s['error']}")

    print("=" * 90)


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Featherweight benchmark runner — runs the same JSONL "
                    "test against multiple models and produces a cross-model "
                    "comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", default="featherweight.jsonl",
                        help="Test JSONL file (default: featherweight.jsonl)")
    parser.add_argument("--models", default="models.json",
                        help="Models config JSON (default: models.json)")
    parser.add_argument("--out", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N questions (debug)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated model names to include")
    parser.add_argument("--prompt-style", default="plain",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help="Prompt format. 'plain' matches the evaluator plugin.")
    args = parser.parse_args()

    # Load test
    try:
        meta, questions = load_test(args.test)
    except FileNotFoundError:
        print(f"ERROR: test file not found: {args.test}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: invalid test file: {e}")
        sys.exit(1)

    if args.limit:
        questions = questions[:args.limit]

    test_name = meta.get("name") or os.path.basename(args.test)
    print(f"Loaded {len(questions)} questions from {args.test}")
    if meta:
        print(f"  Test:    {meta.get('name', '?')}  v{meta.get('version', '?')}")
    print()

    # Load models config
    if not os.path.exists(args.models):
        print(f"ERROR: models config not found: {args.models}")
        print(f"       Copy models.example.json to {args.models} and edit it.")
        sys.exit(1)

    with open(args.models, "r", encoding="utf-8") as f:
        config = json.load(f)

    models = config.get("models", [])
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        models = [m for m in models if m["name"] in wanted]
        missing = wanted - {m["name"] for m in models}
        if missing:
            print(f"WARNING: --only mentioned unknown models: {sorted(missing)}")

    if not models:
        print("ERROR: no models to run")
        sys.exit(1)

    print(f"Running {len(models)} model(s), prompt-style={args.prompt_style}")
    print()

    # Output directory
    run_id = time.strftime("%Y%m%dT%H%M%S")
    run_dir = os.path.join(args.out, f"{run_id}_featherweight")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving results to: {run_dir}/\n")

    summaries = []
    categories_seen = []

    interrupted_globally = False

    for i, cfg in enumerate(models, 1):
        name = cfg["name"]
        print(f"[{i}/{len(models)}] {name}")

        backend = None
        try:
            backend = make_backend(cfg)
        except Exception as e:
            print(f"  Failed to load backend: {e}\n")
            summaries.append({
                "model": name,
                "params": cfg.get("params"),
                "total": 0, "correct": 0, "accuracy": 0,
                "parse_errors": 0, "by_category": {},
                "error": str(e),
            })
            continue

        try:
            result = run_one_model(name, backend, questions,
                                   prompt_style=args.prompt_style)
            # Save per-model JSON immediately
            out_path = os.path.join(run_dir, f"{name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            params = backend.params_count or cfg.get("params")
            summaries.append({
                "model": name,
                "params": params,
                "total": result["total"],
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "parse_errors": result["parse_errors"],
                "elapsed_seconds": result["elapsed_seconds"],
                "by_category": result["by_category"],
            })

            for c in result["by_category"]:
                if c not in categories_seen:
                    categories_seen.append(c)

            print(f"  Score: {result['correct']}/{result['total']}  "
                  f"({result['accuracy'] * 100:.1f}%)  "
                  f"parse_errors={result['parse_errors']}  "
                  f"{result['elapsed_seconds']:.1f}s")
        except KeyboardInterrupt:
            print("\n  Interrupted by user — stopping after this model.")
            interrupted_globally = True
        except Exception as e:
            print(f"  Run failed: {e}")
            import traceback
            traceback.print_exc()
            summaries.append({
                "model": name,
                "params": cfg.get("params"),
                "total": 0, "correct": 0, "accuracy": 0,
                "parse_errors": 0, "by_category": {},
                "error": str(e),
            })
        finally:
            if backend is not None:
                try:
                    backend.cleanup()
                except Exception:
                    pass

        print()

        if interrupted_globally:
            break

    if not summaries:
        print("No models produced results.")
        return

    # Order categories: featherweight natural order if all 10 present, else
    # by first-seen order.
    natural = ["geography", "history", "arithmetic", "science", "language",
               "vocabulary", "common-sense", "logic", "categorization", "reading"]
    if all(c in categories_seen for c in natural):
        categories = natural
    else:
        categories = categories_seen

    # Write summary CSV + JSON
    csv_path = os.path.join(run_dir, "summary.csv")
    write_summary_csv(summaries, csv_path, categories)

    json_path = os.path.join(run_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test": args.test,
            "test_meta": meta,
            "prompt_style": args.prompt_style,
            "timestamp": run_id,
            "interrupted": interrupted_globally,
            "categories": categories,
            "summaries": summaries,
        }, f, ensure_ascii=False, indent=2)

    # Final table
    print_summary_table(summaries, categories, args.prompt_style, test_name)
    print()
    print(f"  Per-model JSONs:  {run_dir}/<model>.json")
    print(f"  Summary CSV:      {csv_path}")
    print(f"  Summary JSON:     {json_path}")


if __name__ == "__main__":
    main()
