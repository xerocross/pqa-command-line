#!/usr/bin/env python3
"""
pfq — Personal FAQ CLI
- Stores items as JSON Lines (JSONL)
- Commands: add, ask, best, search, list, repl
- REPL has tab-completion of known questions (readline-based)

New:
- RapidFuzz-backed fuzzy matching (fallback to difflib)
- `best` subcommand for one-shot, non-interactive "show me the closest answer"
- If invoked as `pqa`, behaves like `pfq best ...`
"""

from __future__ import annotations
import argparse, json, os, sys, uuid, datetime, difflib, tempfile, shutil, readline
from dataclasses import dataclass, asdict
from typing import List, Optional, Iterator, Tuple

# --- Optional fast fuzzy matcher (RapidFuzz). Falls back to difflib. ---
_HAVE_RAPID = False
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAVE_RAPID = True
except Exception:
    _HAVE_RAPID = False

DEFAULT_DIR = os.path.expanduser("~/.personal_faq")
DEFAULT_FILE = os.path.join(DEFAULT_DIR, "faqs.jsonl")

@dataclass
class FAQItem:
    id: str
    question: str
    answer: str
    tags: List[str]
    created_at: str
    updated_at: str

def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat() + "Z"

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
                # Skip malformed lines but continue robustly
                continue
    return items

# --- REPL with robust TAB completion across GNU readline / libedit ---

def init_readline(questions: List[str]) -> None:
    """
    Configure readline for whole-line completion and platform quirks.
    On macOS, Python is often linked against libedit, which needs different bindings.
    """
    # Treat the entire input line as a single "word" so we complete whole questions.
    try:
        readline.set_completer_delims("")  # no delimiters => whole line is a single token
    except Exception:
        pass

    # libedit vs GNU readline binding
    doc = getattr(readline, "__doc__", "") or ""
    if "libedit" in doc:
        # macOS libedit style
        # ^I is TAB; map it to complete
        readline.parse_and_bind("bind ^I rl_complete")
        # libedit doesn't support all 'set' options; keep it minimal
    else:
        # GNU readline (Linux, many builds)
        readline.parse_and_bind("tab: complete")
        # show list without needing double-tab; if unsupported, it’s ignored
        try:
            readline.parse_and_bind("set show-all-if-ambiguous on")
            readline.parse_and_bind("set completion-ignore-case on")
        except Exception:
            pass

    # Basic in-memory completer over the loaded questions
    _q_sorted = sorted(questions, key=str.lower)

    def _complete(text: str, state: int) -> Optional[str]:
        """
        Whole-line prefix completion:
        - 'text' is what readline thinks is the current token (the whole line, since no delimiters)
        - We return full replacements (not suffix-only) to keep libedit happy.
        """
        prefix = (text or "")
        pfx_low = prefix.lower()
        options = [q for q in _q_sorted if q.lower().startswith(pfx_low)]
        if state < len(options):
            return options[state]  # full replacement for the (whole-line) token
        return None

    readline.set_completer(_complete)


def write_all_atomic(path: str, items: List[FAQItem]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="pfq_", suffix=".jsonl", dir=os.path.dirname(path))
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        shutil.move(tmp_path, path)
    finally:
        # If move failed, clean up temp
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
    # Combined text for matching (does NOT change stored data)
    tg = " ".join(it.tags) if it.tags else ""
    return f"{it.question}\n{tg}\n{it.answer}".strip()

def fuzzy_top(items: List[FAQItem], q: str, n: int = 5) -> List[FAQItem]:
    """Return top-N fuzzy matches by question similarity (with substring bump)."""
    if not items:
        return []
    ql = q.lower().strip()

    # Prefer RapidFuzz if available; otherwise use difflib
    if _HAVE_RAPID:
        # We score using weighted ratio over combined fields for better recall.
        choices: List[Tuple[str, int]] = [(_combine_fields_for_match(it), idx) for idx, it in enumerate(items)]
        # extract returns (match_str, score, data)
        ranked = rf_process.extract(ql, choices, scorer=rf_fuzz.WRatio, limit=len(items))
        # ranked is list of tuples (match_str, score, data)
        # Add a small bump if q is a substring of the question field specifically
        scored = []
        for match_str, score, idx in ranked:
            bump = 15 if ql in items[idx].question.lower() else 0
            scored.append((score + bump, items[idx]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:n]]
    else:
        pool = []
        for it in items:
            base = _combine_fields_for_match(it).lower()
            score = difflib.SequenceMatcher(None, ql, base).ratio()
            if ql in it.question.lower():
                score += 0.15
            pool.append((score, it))
        pool.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in pool[:n]]

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
    print(f"Q: {it.question}")
    if show_answer:
        print(f"   A: {it.answer}")
    #if it.tags:
    #    print(f"   tags: {', '.join(it.tags)}")
    #print(f"   id: {it.id}  created: {it.created_at}  updated: {it.updated_at}")

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

def cmd_ask(args: argparse.Namespace) -> int:
    items = read_all(args.file)
    if not items:
        print("No FAQs yet.")
        return 0
    q = " ".join(args.query).strip()
    if not q:
        print("Provide a query.")
        return 1

    it = exact_match(items, q)
    if it:
        print_item(it, show_answer=True)
        return 0

    candidates = fuzzy_top(items, q, n=5)
    if not candidates:
        print("No matches.")
        return 0
    print("Closest matches:")
    for i, c in enumerate(candidates, 1):
        print(f"[{i}] {c.question}")
    try:
        sel = input("Select [1-5] or press Enter to cancel: ").strip()
    except EOFError:
        return 1
    if not sel or not sel.isdigit():
        return 1
    k = int(sel)
    if 1 <= k <= len(candidates):
        print_item(candidates[k-1], show_answer=True)
        return 0
    return 1

def cmd_best(args: argparse.Namespace) -> int:
    """Non-interactive: print the single best fuzzy match for the query."""
    items = read_all(args.file)
    if not items:
        print("No FAQs yet.")
        return 0
    q = " ".join(args.query).strip()
    if not q:
        print("Provide a query.")
        return 1
    # Prefer exact, else top fuzzy
    it = exact_match(items, q)
    if not it:
        top = fuzzy_top(items, q, n=1)
        it = top[0] if top else None
    if not it:
        print("No matches.")
        return 0
    print_item(it, show_answer=not args.questions_only)
    return 0

# --- REPL with tab-completion of known questions ---

class QuestionCompleter:
    def __init__(self, questions: List[str]) -> None:
        self.words = sorted(questions)

    def complete(self, text: str, state: int) -> Optional[str]:
        options = [w for w in self.words if w.lower().startswith((text or "").lower())]
        if state < len(options):
            return options[state]
        return None

def repl(path: str) -> int:
    items = read_all(path)
    if not items:
        print("No FAQs yet. Use `pfq add` first.")
        return 0

    print("pfq REPL — type your question; TAB to complete; :q to quit; :help for commands")
    print(f"(Loaded {len(items)} Qs from {path})")
    init_readline([it.question for it in items])

    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line is None:
            continue
        line = line.strip()
        if not line:
            continue

        if line in (":q", ":quit", ":exit"):
            break
        if line == ":help":
            print(":help — show this help")
            print(":list — list all questions")
            print(":reload — reload data file")
            print(":add — add a new FAQ interactively")
            print(":q — quit")
            continue
        if line == ":list":
            for i, it in enumerate(items, 1):
                print(f"[{i}] {it.question}")
            continue
        if line == ":reload":
            items = read_all(path)
            init_readline([it.question for it in items])
            print(f"Reloaded {len(items)} items.")
            continue
        if line == ":add":
            q = input("Question: ").strip()
            print("Answer (finish with EOF):")
            chunks: List[str] = []
            try:
                while True:
                    chunks.append(input())
            except EOFError:
                pass
            a = "\n".join(chunks).strip()
            add_item(path, q, a, [])
            items = read_all(path)
            init_readline([it.question for it in items])
            print("Added.")
            continue

        # normal question path
        em = exact_match(items, line)
        if em:
            print_item(em, show_answer=True)
            continue

        guesses = fuzzy_top(items, line, n=5)
        if not guesses:
            print("No matches.")
            continue
        print("Closest:")
        for i, g in enumerate(guesses, 1):
            print(f"[{i}] {g.question}")
        choice = input("Pick [1-5] or Enter to skip: ").strip()
        if choice.isdigit():
            ix = int(choice) - 1
            if 0 <= ix < len(guesses):
                print_item(guesses[ix], show_answer=True)

    return 0

def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog or "pfq", description="Personal FAQ (JSONL-backed)")
    p.add_argument("--file", default=os.environ.get("PFQ_FILE", DEFAULT_FILE),
                   help=f"Path to JSONL store (default: {DEFAULT_FILE})")
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

    sp = sub.add_parser("ask", help="Ask a question (exact/fuzzy, interactive)")
    sp.add_argument("query", nargs=argparse.REMAINDER, help="Your question")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("best", help="Non-interactive: print best fuzzy match")
    sp.add_argument("query", nargs=argparse.REMAINDER, help="Your question")
    sp.add_argument("--questions-only", action="store_true", help="Hide answer")
    sp.set_defaults(func=cmd_best)

    sp = sub.add_parser("repl", help="Interactive mode with TAB completion")
    sp.set_defaults(func=lambda a: repl(a.file))

    return p

def _invoked_as_pqa() -> bool:
    base = os.path.basename(sys.argv[0]).lower()
    return base in {"pqa", "pfqa"}

def main(argv: List[str]) -> int:
    # If invoked as `pqa`, treat as `pfq best <args>`
    if _invoked_as_pqa():
        parser = build_parser(prog="pqa")
        # Shim: if user passes nothing, show help; otherwise force 'best'
        # if argv and argv[0] in {"-h", "--help"}:
        #     # Build a temp parser to show 'best' usage succinctly
        #     print("Usage: pqa [--file PATH] <your question...>\n       pqa --help")
        #     return 0
        # Prepend the implicit 'best' subcommand
        # argv = ["best"] + argv
        args = parser.parse_args(argv)
        ensure_store(args.file)
        return args.func(args)

    parser = build_parser(prog="pfq")
    args = parser.parse_args(argv)
    ensure_store(args.file)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
