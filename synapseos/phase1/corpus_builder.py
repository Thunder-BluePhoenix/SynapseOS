"""
Training corpus builder for the Mamba Sequence Engine.

Collects and cleans domain text from Frappe/ERPNext apps:
  - Python source files (controllers, utilities, hooks)
  - DocType JSON definitions (converted to structured text)
  - Markdown documentation files

Output: plain text corpus files in data/corpus/ ready for tokenization.
Target: ~500M tokens total for Mamba LoRA fine-tuning.
"""

import json
import re
from pathlib import Path


# Extensions to collect source from
SOURCE_EXTENSIONS = {".py", ".js"}
DOC_EXTENSIONS = {".md", ".txt", ".rst"}

# Files/dirs to skip
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".tox", "dist", "build",
    "test", "tests", "public", "assets",
}
SKIP_FILES = {"setup.py", "conftest.py"}

# Minimum characters for a file to be included
MIN_FILE_CHARS = 200


class CorpusBuilder:
    def __init__(self, apps_dir: str, output_dir: str, app_filter: list[str] | None = None):
        self.apps_dir = Path(apps_dir)
        self.output_dir = Path(output_dir)
        self.app_filter = app_filter
        self.stats: dict[str, int] = {
            "files_processed": 0,
            "files_skipped": 0,
            "total_chars": 0,
            "python_chars": 0,
            "json_chars": 0,
            "doc_chars": 0,
        }

    def build(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apps = self._get_apps()

        # Per-app files
        for app_path in apps:
            print(f"  Processing: {app_path.name}")
            self._process_app(app_path)

        # Merge all per-app files into one combined corpus file
        self._merge_corpus(apps)

        self._write_stats()
        self._print_summary()

    def _merge_corpus(self, apps: list[Path]) -> None:
        merged_path = self.output_dir / "corpus_all.txt"
        with open(merged_path, "w") as merged:
            for app_path in apps:
                for suffix in ("python", "doctype", "docs"):
                    part = self.output_dir / f"{app_path.name}_{suffix}.txt"
                    if part.exists() and part.stat().st_size > 0:
                        merged.write(f"\n\n### APP: {app_path.name} / {suffix} ###\n\n")
                        merged.write(part.read_text(encoding="utf-8", errors="ignore"))
        print(f"\nMerged corpus → {merged_path}")

    def _get_apps(self) -> list[Path]:
        apps = []
        for item in sorted(self.apps_dir.iterdir()):
            if not item.is_dir():
                continue
            if self.app_filter and item.name not in self.app_filter:
                continue
            apps.append(item)
        return apps

    def _process_app(self, app_path: Path) -> None:
        app_name = app_path.name
        py_out = self.output_dir / f"{app_name}_python.txt"
        json_out = self.output_dir / f"{app_name}_doctype.txt"
        doc_out = self.output_dir / f"{app_name}_docs.txt"

        with open(py_out, "w") as py_f, open(json_out, "w") as json_f, open(doc_out, "w") as doc_f:
            for file_path in sorted(app_path.rglob("*")):
                if not file_path.is_file():
                    continue
                if self._should_skip(file_path):
                    self.stats["files_skipped"] += 1
                    continue

                suffix = file_path.suffix.lower()

                if suffix in SOURCE_EXTENSIONS:
                    text = self._process_source_file(file_path)
                    if text:
                        py_f.write(text)
                        py_f.write("\n\n")
                        self.stats["python_chars"] += len(text)
                        self.stats["files_processed"] += 1

                elif suffix == ".json" and "doctype" in str(file_path):
                    text = self._process_doctype_json(file_path)
                    if text:
                        json_f.write(text)
                        json_f.write("\n\n")
                        self.stats["json_chars"] += len(text)
                        self.stats["files_processed"] += 1

                elif suffix in DOC_EXTENSIONS:
                    text = self._process_doc_file(file_path)
                    if text:
                        doc_f.write(text)
                        doc_f.write("\n\n")
                        self.stats["doc_chars"] += len(text)
                        self.stats["files_processed"] += 1

        self.stats["total_chars"] = (
            self.stats["python_chars"] + self.stats["json_chars"] + self.stats["doc_chars"]
        )

    def _should_skip(self, file_path: Path) -> bool:
        for part in file_path.parts:
            if part in SKIP_DIRS:
                return True
        if file_path.name in SKIP_FILES:
            return True
        return False

    def _process_source_file(self, file_path: Path) -> str | None:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

        if len(text) < MIN_FILE_CHARS:
            return None

        # Strip blank lines and normalize whitespace
        lines = [line.rstrip() for line in text.splitlines()]
        cleaned = "\n".join(lines)

        # Add file header as context marker
        return f"# FILE: {file_path}\n{cleaned}"

    def _process_doctype_json(self, file_path: Path) -> str | None:
        try:
            with open(file_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if not isinstance(data, dict) or data.get("doctype") != "DocType":
            return None

        name = data.get("name", "")
        module = data.get("module", "")
        doc_type = data.get("document_type", "")
        is_sub = bool(data.get("istable", 0))
        is_submit = bool(data.get("is_submittable", 0))

        lines = [
            f"# DOCTYPE: {name}",
            f"# Module: {module}  Type: {doc_type}  Submittable: {is_submit}  ChildTable: {is_sub}",
            "",
            "## Fields",
        ]

        for field in data.get("fields", []):
            ft = field.get("fieldtype", "")
            fn = field.get("fieldname", "")
            label = field.get("label", fn)
            opts = field.get("options", "")
            reqd = "required" if field.get("reqd") else ""
            unique = "unique" if field.get("unique") else ""
            flags = ", ".join(f for f in [reqd, unique] if f)
            if ft in {"Section Break", "Column Break", "Tab Break", "HTML"}:
                continue
            line = f"  {fn} ({ft}): {label}"
            if opts:
                line += f" → {opts}"
            if flags:
                line += f" [{flags}]"
            if field.get("mandatory_depends_on"):
                line += f" [mandatory_if: {field['mandatory_depends_on']}]"
            lines.append(line)

        permissions = data.get("permissions", [])
        if permissions:
            lines.append("")
            lines.append("## Permissions")
            for perm in permissions:
                role = perm.get("role", "")
                actions = [a for a in ["read", "write", "create", "delete", "submit", "cancel"] if perm.get(a)]
                lines.append(f"  {role}: {', '.join(actions)}")

        text = "\n".join(lines)
        if len(text) < MIN_FILE_CHARS:
            return None
        return text

    def _process_doc_file(self, file_path: Path) -> str | None:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

        if len(text) < MIN_FILE_CHARS:
            return None

        # Remove HTML tags if any crept into markdown
        text = re.sub(r"<[^>]+>", "", text)
        return f"# DOC: {file_path}\n{text.strip()}"

    def _write_stats(self) -> None:
        stats_path = self.output_dir / "corpus_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

    def _print_summary(self) -> None:
        total_mb = self.stats["total_chars"] / 1_000_000
        # Rough token estimate: ~4 chars per token
        est_tokens_m = self.stats["total_chars"] / 4 / 1_000_000
        print("\n--- Corpus Stats ---")
        print(f"  Files processed : {self.stats['files_processed']}")
        print(f"  Files skipped   : {self.stats['files_skipped']}")
        print(f"  Total size      : {total_mb:.1f} MB")
        print(f"  Est. tokens     : ~{est_tokens_m:.1f}M")
        print(f"  Python          : {self.stats['python_chars'] / 1_000_000:.1f} MB")
        print(f"  DocType JSON    : {self.stats['json_chars'] / 1_000_000:.1f} MB")
        print(f"  Documentation   : {self.stats['doc_chars'] / 1_000_000:.1f} MB")
        print(f"\n  Output → {self.output_dir}")
        if est_tokens_m < 500:
            print(f"\n  NOTE: {est_tokens_m:.0f}M tokens collected — target is 500M.")
            print("  Consider adding: external docs, issue tracker exports, forum posts.")
