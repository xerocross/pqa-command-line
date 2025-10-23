# pqa (Personal FAQ CLI)

**pqa** is a lightweight command‑line tool for managing a personal FAQ / knowledge base. It stores each Q/A as a single JSON Lines (JSONL) record and gives you fast recall via fuzzy matching and an optional interactive REPL.

---

## ✨ Features

- **Plain JSONL storage** (human‑readable, easy to back up)
- **Configurable data path** via `~/.config/pqa/config.txt`
- **Fuzzy search & ranking** (uses RapidFuzz if installed; falls back gracefully)
- **Intent‑aware scoring** that favors “how to …” queries
- **Interactive ask mode** with `prompt_toolkit` (optional)
- **Atomic writes** to avoid data corruption

---

## 🧩 Installation

```bash
# Clone
git clone https://github.com/yourusername/pqa.git
cd pqa

# (optional) venv
python3 -m venv .venv
source .venv/bin/activate

# Optional (recommended) extras
pip install rapidfuzz prompt_toolkit

# Make executable (if needed)
chmod +x pqa.py

# Put on PATH (example)
ln -s "$(pwd)/pqa.py" ~/.local/bin/pqa
```

> **Python**: 3.9+ recommended.

---

## ⚙️ Configuration

Create a config file at:

```
~/.config/pqa/config.txt
```

with at least:

```
data_file_path=~/Documents/pqa/faqs.jsonl
```

You can override the store location per‑command or via an environment variable:

- CLI override:
  ```bash
  pqa --file /custom/path/faqs.jsonl list
  ```
- Environment variable:
  ```bash
  export PQA_FILE=/custom/path/faqs.jsonl
  ```

**Precedence**: `--file` > `PQA_FILE` env var > config file.

---

## 🚀 Usage

### Add a new entry
```bash
pqa add -q "How to restart nginx?" -a "sudo systemctl restart nginx"
```

(Or interactively: `pqa add` and follow prompts.)

### Ask a question (non‑interactive)
```bash
pqa ask "how to restart nginx"
```
Prints the single best match and its answer.

### Interactive ask mode
```bash
pqa ask -i
```
Type to fuzzy search; Enter shows the closest match. `Ctrl‑C`/`Ctrl‑D` to exit.

### Search across questions, answers, and tags
```bash
pqa search nginx
pqa search --questions-only backup
```

### List all questions
```bash
pqa list
```

---

## 🧠 Ranking (intent‑aware)

- Detects **how‑to phrasing** in your query and weights toward matching the **question** field.
- Uses **token‑set** similarity + overall text similarity.
- Falls back to a simple Jaccard/difflib approach if RapidFuzz isn’t installed.

Result: queries like *“how to add ssh key”* can match *“How to set up SSH keys for GitHub”* even when phrased differently.

---

## 🧰 Storage & Safety

- **Format**: JSON Lines (`faqs.jsonl`), one record per line.
- **Atomic writes**: data is written to a temp file in the same directory, then moved into place.
- **No external DB required**; trivial to version with Git and back up.

---

## 🔧 Troubleshooting

- **Config not found**: ensure `~/.config/pqa/config.txt` exists and contains a valid `data_file_path=` line.
- **Permissions**: make sure the parent directory of `data_file_path` is writable.
- **Interactive mode errors**: install `prompt_toolkit` → `pip install prompt_toolkit`.
- **Fuzzy quality**: install `rapidfuzz` → `pip install rapidfuzz`.

---

## 🧪 Quick smoke test

```bash
# Use a temporary store
PQA_FILE=/tmp/pqa_demo.jsonl pqa add -q "How to exit vim?" -a ":q"
PQA_FILE=/tmp/pqa_demo.jsonl pqa ask "exit vim"
PQA_FILE=/tmp/pqa_demo.jsonl pqa list
```

---

## 📦 Backup & Sync Tips

- Commit the JSONL to Git for history and diff‑ability.
- Keep your `~/.config/pqa/config.txt` under a dotfiles manager.
- Periodically export to Markdown if you want a browsable snapshot.

---

## 📝 License

MIT (or your preferred license). See `LICENSE` in this repository.

---

## 🙌 Contributions

Issues and PRs are welcome. Please keep changes small and well‑scoped; include a brief description, repro steps (if applicable), and before/after behavior.

