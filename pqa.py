#!/usr/bin/env python3
"""
pqa — Personal FAQ CLI
- Stores items as JSON Lines (JSONL)
- Commands: add, ask, best (DEPRECATED), search, list, config
- Interactive ask mode via `pqa ask -i` (prompt_toolkit fuzzy REPL if available)

Matching:
- RapidFuzz-backed fuzzy matching (fallback to difflib)
- Non-interactive `ask` prints the single best match (previously `best`)
- `best` is deprecated but still works (emits a warning)
- Intent-aware scoring: 'how-to' phrasing biases toward question-field coverage
"""

from __future__ import annotations
import argparse, json, os, sys, uuid, datetime, difflib, tempfile, shutil, re
from dataclasses import dataclass, asdict
from typing import List, Optional, Set
from pathlib import Path  # <-- config path handling

# --- Optional fast fuzzy matcher (RapidFuzz). Falls back to difflib. ---
_HAVE_RAPID = False
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAVE_RAPID = True
except Exception:
    _HAVE_RAPID = False

VERSION = "0.1.1"

# ---------------------------------------------------------------------
# Storage location resolution (CLI --file > env PQA_FILE > config file)
# Config file: ~/.config/pqa/config.txt with key `data_file_path=...`
# If neither CLI nor env override is set, config is REQUIRED (except for `config --show`).
# ---------------------------------------------------------------------
CONFIG_PATH = Path.home() / ".config" / "pqa" / "config.txt"

def _parse_kv_file(path: Path) -> dict:
    """Lenient key=value parser (ignores blanks, comments, and malformed lines)."""
    cfg = {}
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg

def _load_config_data_path() -> Path:
    """Strict load for runtime: exit with message if config missing/bad."""
    if not CONFIG_PATH.exists():
        sys.exit(f"Config file not found: {CONFIG_PATH}\n"
                 f"Create it with a line like:\n"
                 f"  data_file_path=~/Documents/pqa/faqs.jsonl")
    try:
        cfg = _parse_kv_file(CONFIG_PATH)
    except Exception as e:
        sys.exit(f"Failed to read config file {CONFIG_PATH}: {e}")
    val = cfg.get("data_file_path")
    if not val:
        sys.exit(f"Missing 'data_file_path' in {CONFIG_PATH}")
    p = Path(os.path.expandvars(os.path.expanduser(val))).resolve()
    parent = p.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            sys.exit(f"Cannot create parent directory for data file {p}: {e}")
    return p

def resolve_store_path(cli_file: Optional[str]) -> str:
    """
    Choose the JSONL store path with precedence:
      1) CLI --file
      2) PQA_FILE env var
      3) Config file (required if 1/2 not set)
    """
    if cli_file:
        return os.path.expandvars(os.path.expanduser(cli_file))
    env = os.environ.get("PQA_FILE")
    if env:
        return os.path.expandvars(os.path.expanduser(env))
    return str(_load_config_data_path())

def inspect_config(cli_file: Optional[str]) -> dict:
    """
    Non-strict inspection for `pqa config --show`.
    Does NOT exit on errors; returns a dict describing what was found.
    """
    info = {
        "version": VERSION,
        "config_path": str(CONFIG_PATH),
        "config_exists": CONFIG_PATH.exists(),
        "env_PQA_FILE": os.environ.get("PQA_FILE"),
        "cli_file": cli_file,
        "config_data_file_path_raw": None,
        "config_data_file_path_resolved": None,
        "config_error": None,
        "chosen_source": None,     # "cli" | "env" | "config" | None
        "chosen_path": None,
    }

    # If config exists, try to parse it
    if info["config_exists"]:
        try:
            cfg = _parse_kv_file(CONFIG_PATH)
            raw = cfg.get("data_file_path")
            info["config_data_file_path_raw"] = raw
            if raw:
                info["config_data_file_path_resolved"] = str(
                    Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
                )
        except Exception as e:
            info["config_error"] = f"{type(e).__name__}: {e}"

    # Choose source (without strict validation or mkdirs)
    if cli_file:
        info["chosen_source"] = "cli"
        info["chosen_path"] = os.path.expandvars(os.path.expanduser(cli_file))
    elif info["env_PQA_FILE"]:
        info["chosen_source"] = "env"
        info["chosen_path"] = os.path.expandvars(os.path.expanduser(info["env_PQA_FILE"]))
    elif info["config_data_file_path_resolved"]:
        info["chosen_source"] = "config"
        info["chosen_path"] = info["config_data_file_path_resolved"]
    else:
        info["chosen_source"] = None
        info["chosen_path"] = None

    return info

@dataclass
class FAQItem:
    id: str
    question: str
    answer: str
    tags: List[str]
    created_at: str
    updated_at: str

def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def ensure_store(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8"):
            pass

def read_all(path: str) -> List[FAQItem]:
    items: List[FAQItem] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(FAQItem(**{
                    "id": obj.get("id", str(uuid.uuid4())),
                    "question": obj["question"],
                    "answer": obj["answer"],
                    "tags": obj.get("tags", []),
                    "created_at": obj.get("created_at", now_iso()),
                    "updated_at": obj.get("updated_at", obj.get("created_at", now_iso())),
                }))
            except Exception:
                continue
    return items

def write_all_atomic(path: str, items: List[FAQItem]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="pqa_", suffix=".jsonl", dir=os.path.dirname(path))
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        shutil.move(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def add_item(path: str, question: str, answer: str, tags: List[str]) -> FAQItem:
    ensure_store(path)
    items = read_all(path)
    now = now_iso()
    new = FAQItem(
        id=str(uuid.uuid4()),
        question=question.strip(),
        answer=answer.strip(),
        tags=[t.strip() for t in tags if t.strip()],
        created_at=now,
        updated_at=now,
    )
    items.append(new)
    write_all_atomic(path, items)
    return new

def exact_match(items: List[FAQItem], q: str) -> Optional[FAQItem]:
    qnorm = q.strip().lower()
    for it in items:
        if it.question.strip().lower() == qnorm:
            return it
    return None

def _combine_fields_for_match(it: FAQItem) -> str:
    tg = " ".join(it.tags) if it.tags else ""
    return f"{it.question}\n{tg}\n{it.answer}".strip()

# ------------------------
# Intent-aware scoring
# ------------------------

_STOPWORDS: Set[str] = {
    "a","an","the","do","does","did","to","of","in","on","for","with","and","or","is","are","am",
    "be","been","being","i","you","we","they","he","she","it","my","your","our","their","me","him",
    "her","how","what","when","where","why","which","that","this","these","those","can","could",
    "should","would","will","shall","from","by","as","at","into","about","than","then","up","down"
}

_HOWTO_PAT = re.compile(
    r"^\s*(?:how\s+to|how\s+do\s+i|how\s+do\s+you|how\s+can\s+i|how\s+can\s+you|what'?s\s+the\s+way\s+to)\b",
    re.IGNORECASE
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def _simple_stem(tok: str) -> str:
    # very light stemming to align close variants without extra deps
    t = tok.lower()
    for suf in ("ing","ed","es","s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t

def _content_tokens(text: str) -> Set[str]:
    toks = [ _simple_stem(m.group(0)) for m in _TOKEN_RE.finditer(text.lower()) ]
    return {t for t in toks if t and t not in _STOPWORDS}

def _is_howto_query(q: str) -> bool:
    return bool(_HOWTO_PAT.search(q))

def _token_set_ratio(a: str, b: str) -> float:
    """0..1 similarity between token sets (RapidFuzz when available, else Jaccard)."""
    if _HAVE_RAPID:
        # token_set_ratio returns 0..100
        return rf_fuzz.token_set_ratio(a, b) / 100.0
    # Fallback Jaccard over content tokens
    A, B = _content_tokens(a), _content_tokens(b)
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def _wr_or_ratio(a: str, b: str) -> float:
    """WRatio (0..1) with difflib fallback."""
    if _HAVE_RAPID:
        return rf_fuzz.WRatio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _coverage(query_tokens: Set[str], text: str) -> float:
    """Fraction of query content tokens appearing in text (0..1)."""
    if not query_tokens:
        return 0.0
    T = _content_tokens(text)
    return len(query_tokens & T) / max(1, len(query_tokens))

def _intent_aware_score(it: FAQItem, q: str) -> float:
    """
    Composite score tailored for Q/A retrieval.
    Emphasizes question-field alignment, especially for 'how-to' phrasing.
    Returns a float; higher is better.
    """
    howto = _is_howto_query(q)
    q_tokens = _content_tokens(q)

    # Core components
    q_coverage = _coverage(q_tokens, it.question)            # 0..1
    q_token_sim = _token_set_ratio(q, it.question)           # 0..1
    combined_sim = _wr_or_ratio(q, _combine_fields_for_match(it))  # 0..1

    # Weighting: lean harder on question alignment for how-to
    if howto:
        # favor questions that contain the "action" words; minimize answer-lures
        score = (0.55 * q_coverage) + (0.30 * q_token_sim) + (0.15 * combined_sim)
    else:
        score = (0.40 * q_coverage) + (0.30 * q_token_sim) + (0.30 * combined_sim)

    # Small boost if the item's question starts with a how-to template when query is how-to
    if howto and it.question.lower().startswith(("how to","how do i","how do you","how can i","how can you")):
        score += 0.05

    return score

# ------------------------
# Matchers (using intent-aware score)
# ------------------------

def _best_match(items: List[FAQItem], q: str) -> Optional[FAQItem]:
    """Single best match with intent-aware scoring (how-to bias)."""
    if not items:
        return None
    ql = q.strip()
    winner, win_score = None, float("-inf")

    # Try RapidFuzz top-k over questions to reduce work (if available), then rescore.
    candidates: List[FAQItem]
    if _HAVE_RAPID:
        qtexts = [it.question for it in items]
        prelim = rf_process.extract(ql, qtexts, scorer=rf_fuzz.WRatio, limit=min(25, len(items)))
        idxs = [i for (_, _, i) in prelim]
        candidates = [items[i] for i in idxs]
    else:
        candidates = items

    for it in candidates:
        s = _intent_aware_score(it, ql)
        if s > win_score:
            winner, win_score = it, s
    return winner

def fuzzy_top(items: List[FAQItem], q: str, n: int = 5) -> List[FAQItem]:
    """Return top-N matches using intent-aware score."""
    if not items:
        return []
    ql = q.strip()
    scored = [(_intent_aware_score(it, ql), it) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:n]]

def search_items(items: List[FAQItem], term: str, limit: int = 20) -> List[FAQItem]:
    term_l = term.strip().lower()
    ranked = fuzzy_top(items, term_l, n=len(items))
    out: List[FAQItem] = []
    for it in ranked:
        if (term_l in it.question.lower()
            or term_l in it.answer.lower()
            or any(term_l in t.lower() for t in it.tags)):
            out.append(it)
            if len(out) >= limit:
                break
    return out or ranked[: min(limit, 10)]

def print_item(it: FAQItem, show_answer: bool = True, idx: Optional[int] = None) -> None:
    prefix = f"[{idx}] " if idx is not None else ""
    print(f"{prefix}Q: {it.question}")
    if show_answer:
        print(f"   A: {it.answer}")

# --- Core "best answer" routine now shared by ask(non-interactive) and best(deprecated) ---

def answer_best(path: str, query: str, *, questions_only: bool = False) -> int:
    items = read_all(path)
    if not items:
        print("No FAQs yet.")
        return 0
    q = (query or "").strip()
    if not q:
        print("Provide a query.", file=sys.stderr)
        return 1
    it = exact_match(items, q) or _best_match(items, q)
    if not it:
        print("No matches.")
        return 0
    print_item(it, show_answer=not questions_only)
    return 0

# --- Commands ---

def cmd_add(args: argparse.Namespace) -> int:
    path = args.file
    ensure_store(path)
    question = (args.question or "").strip() or input("Question: ").strip()

    answer = args.answer
    if not answer:
        print("Enter answer. Finish with EOF (Ctrl+D on *nix, Ctrl+Z on Windows).")
        print("-" * 40)
        chunks: List[str] = []
        try:
            while True:
                chunks.append(input())
        except EOFError:
            pass
        answer = "\n".join(chunks).strip()

    tags = args.tags or []
    item = add_item(path, question, answer, tags)
    print("Added:")
    print_item(item)
    return 0

def cmd_list(args: argparse.Namespace) -> int:
    items = read_all(args.file)
    if not items:
        print("No FAQs yet.")
        return 0
    for i, it in enumerate(items, 1):
        print_item(it, show_answer=False, idx=i)
    return 0

def cmd_search(args: argparse.Namespace) -> int:
    items = read_all(args.file)
    if not items:
        print("No FAQs yet.")
        return 0
    hits = search_items(items, args.term, limit=args.limit)
    if not hits:
        print("No matches.")
        return 0
    for i, it in enumerate(hits, 1):
        print_item(it, show_answer=not args.questions_only, idx=i)
    return 0

def interactive_ask(path: str, limit: int = 5) -> int:
    """Fuzzy REPL using prompt_toolkit if present; graceful guidance otherwise."""
    items = read_all(path)
    if not items:
        print("No FAQs yet. Use `pqa add` first.")
        return 0
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
    except Exception:
        print("Interactive mode requires 'prompt_toolkit'. Try: pip install prompt_toolkit", file=sys.stderr)
        return 2

    questions = [it.question for it in items]
    completer = FuzzyCompleter(WordCompleter(questions, ignore_case=True))
    print(f"pqa ask (interactive) — type to fuzzy search • Enter shows best • Ctrl-C/Ctrl-D to exit")
    print(f"(Loaded {len(items)} Qs from {path})")

    while True:
        try:
            q = prompt("ask> ", completer=completer, complete_while_typing=True)
        except (KeyboardInterrupt, EOFError):
            print()
            break
        q = (q or "").strip()
        if not q:
            continue

        it = exact_match(items, q) or _best_match(items, q)
        if not it:
            print("No matches.")
            continue

        picks = fuzzy_top(items, q, n=max(1, min(limit, 10)))
        if picks:
            print("Closest matches:")
            for i, c in enumerate(picks, 1):
                print(f"[{i}] {c.question}")
            print()
            print_item(picks[0], show_answer=True)
        else:
            print_item(it, show_answer=True)
    return 0

def cmd_ask(args: argparse.Namespace) -> int:
    # Keep interactive path exactly as-is
    if args.interactive:
        return interactive_ask(args.file, limit=args.limit)

    # Non-interactive: behave like old `best` with intent-aware scoring
    q = " ".join(args.query).strip()
    return answer_best(args.file, q, questions_only=False)

def cmd_best(args: argparse.Namespace) -> int:
    """DEPRECATED: Non-interactive single best match (kept for compatibility)."""
    q = " ".join(args.query).strip()
    print(
        "warning: 'pqa best' is deprecated and will be removed in a future release. "
        "Use: pqa ask <query>",
        file=sys.stderr,
    )
    return answer_best(args.file, q, questions_only=args.questions_only)

def cmd_config(args: argparse.Namespace) -> int:
    """`pqa config --show` prints resolved config and source without exiting on errors."""
    info = inspect_config(args.file)
    # Pretty, stable output (easy to parse if needed)
    print(f"pqa {info['version']}")
    print(f"Config file: {info['config_path']}  (exists: {info['config_exists']})")
    if info["config_error"]:
        print(f"Config error: {info['config_error']}")
    print(f"CLI --file: {info['cli_file']}")
    print(f"Env PQA_FILE: {info['env_PQA_FILE']}")
    print(f"Config data_file_path (raw): {info['config_data_file_path_raw']}")
    print(f"Config data_file_path (resolved): {info['config_data_file_path_resolved']}")
    print(f"Chosen source: {info['chosen_source']}")
    print(f"Chosen path: {info['chosen_path']}")
    return 0

# --- CLI ---

def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog or "pqa", description="Personal FAQ (JSONL-backed)")
    # global --version
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    # default=None ensures we can detect absence and then require config
    p.add_argument(
        "--file",
        default=None,
        help="Path to JSONL store. Precedence: --file > PQA_FILE env > config (~/.config/pqa/config.txt)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("add", help="Add a new Q/A")
    sp.add_argument("-q", "--question", help="Question text")
    sp.add_argument("-a", "--answer", help="Answer text (omit to enter multi-line)")
    sp.add_argument("-t", "--tags", nargs="*", default=[], help="Tags")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("list", help="List questions")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("search", help="Search questions/answers/tags")
    sp.add_argument("term", help="Search term")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--questions-only", action="store_true", help="Hide answers in output")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("ask", help="Ask a question (use -i for interactive REPL; non-interactive prints single best match)")
    sp.add_argument("-i", "--interactive", action="store_true",
                    help="Interactive fuzzy mode (prompt_toolkit if available)")
    sp.add_argument("--limit", type=int, default=5,
                    help="Interactive mode: how many close matches to show (default: 5)")
    sp.add_argument("query", nargs=argparse.REMAINDER, help="Your question (omit with -i)")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("best", help="DEPRECATED: Non-interactive single best match (use 'pqa ask <query>')")
    sp.add_argument("query", nargs=argparse.REMAINDER, help="Your question")
    sp.add_argument("--questions-only", action="store_true", help="Hide answer")
    sp.set_defaults(func=cmd_best)

    sp = sub.add_parser("config", help="Inspect configuration")
    sp.add_argument("--show", action="store_true", help="Print resolved paths and sources")
    sp.set_defaults(func=cmd_config)

    # Compatibility shim: `pqa repl` behaves like `pqa ask -i`
    sp = sub.add_parser("repl", help="(deprecated) use: pqa ask -i")
    sp.set_defaults(func=lambda a: interactive_ask(a.file))

    return p

def main(argv: List[str]) -> int:
    parser = build_parser(prog="pqa")
    args = parser.parse_args(argv)

    if args.cmd == "config":
        # `config --show` should not require the store to exist or config to be valid.
        return args.func(args)

    # Resolve the store path using precedence (CLI > env > config)
    resolved = resolve_store_path(args.file)
    args.file = resolved

    ensure_store(args.file)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
