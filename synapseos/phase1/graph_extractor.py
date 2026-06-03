"""
Frappe knowledge graph extractor.

Scans all Frappe app directories, parses DocType JSON files, and builds a
NetworkX DiGraph representing the full domain knowledge: entities, fields,
relationships, permissions, and workflow transitions.

Usage:
    from synapseos.phase1.graph_extractor import FrappeGraphExtractor
    extractor = FrappeGraphExtractor(apps_dir="/path/to/bench/apps")
    G = extractor.build()
    extractor.save("data/graph/knowledge_graph.json")
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import networkx as nx


LINK_FIELD_TYPES = {"Link", "Dynamic Link"}
CHILD_TABLE_TYPES = {"Table", "Table MultiSelect"}

# Hooks that define DocType lifecycle rules
DOC_EVENT_HOOKS = {
    "validate", "before_save", "after_insert", "on_update",
    "before_submit", "on_submit", "before_cancel", "on_cancel",
    "on_trash", "before_naming", "autoname",
}


class FrappeGraphExtractor:
    def __init__(self, apps_dir: str, app_filter: list[str] | None = None):
        self.apps_dir = Path(apps_dir)
        self.app_filter = app_filter  # None = all apps
        self.G = nx.DiGraph()
        self.stats: dict[str, int] = {
            "apps": 0,
            "doctypes": 0,
            "fields": 0,
            "link_edges": 0,
            "child_table_edges": 0,
            "workflow_edges": 0,
            "hook_rules": 0,
        }

    def build(self) -> nx.DiGraph:
        apps = self._get_apps()
        for app_path in apps:
            self._process_app(app_path)
        for app_path in apps:
            self._process_hooks(app_path)
        return self.G

    def save(self, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = nx.node_link_data(self.G)
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)

        stats_path = out.parent / "graph_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        print(f"Graph saved → {out}")
        print(f"Stats saved → {stats_path}")
        self._print_stats()

    def _get_apps(self) -> list[Path]:
        if not self.apps_dir.exists():
            raise FileNotFoundError(f"Apps directory not found: {self.apps_dir}")

        apps = []
        for item in sorted(self.apps_dir.iterdir()):
            if not item.is_dir():
                continue
            if self.app_filter and item.name not in self.app_filter:
                continue
            apps.append(item)

        self.stats["apps"] = len(apps)
        return apps

    def _process_app(self, app_path: Path) -> None:
        for json_file in app_path.rglob("*.json"):
            # Only DocType definition files (named <doctype_name>.json inside a doctype folder)
            if "doctype" not in json_file.parts:
                continue
            if json_file.stem.replace(" ", "_").lower() not in json_file.parent.name.replace(" ", "_").lower():
                # Skip files that aren't the primary doctype definition
                if json_file.stem != json_file.parent.name:
                    continue
            self._parse_doctype_json(json_file, app_path.name)

    def _parse_doctype_json(self, json_file: Path, app_name: str) -> None:
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if data.get("doctype") != "DocType":
            return

        name = data.get("name")
        if not name:
            return

        # Add node for this DocType
        self.G.add_node(
            name,
            type="doctype",
            app=app_name,
            module=data.get("module", ""),
            document_type=data.get("document_type", ""),
            is_submittable=bool(data.get("is_submittable", 0)),
            is_child_table=bool(data.get("istable", 0)),
            autoname=data.get("autoname", ""),
            description=data.get("description", ""),
        )
        self.stats["doctypes"] += 1

        # Process fields
        for field in data.get("fields", []):
            self._process_field(name, field)

        # Process permissions → store as node attribute
        roles = [p.get("role", "") for p in data.get("permissions", []) if p.get("role")]
        if roles:
            self.G.nodes[name]["roles"] = list(set(roles))

        # Process workflows embedded in the doctype (rare, but exists)
        # Main workflow parsing is done via workflow doctype JSON separately

    def _process_field(self, doctype: str, field: dict[str, Any]) -> None:
        fieldtype = field.get("fieldtype", "")
        fieldname = field.get("fieldname", "")
        if not fieldname:
            return

        self.stats["fields"] += 1

        if fieldtype in LINK_FIELD_TYPES:
            target = field.get("options", "")
            if target and fieldtype == "Link":
                # Ensure target node exists (may be parsed later)
                if not self.G.has_node(target):
                    self.G.add_node(target, type="doctype")

                self.G.add_edge(
                    doctype,
                    target,
                    relation="link",
                    fieldname=fieldname,
                    label=field.get("label", fieldname),
                    required=bool(field.get("reqd", 0)),
                )
                self.stats["link_edges"] += 1

            elif fieldtype == "Dynamic Link":
                # Dynamic link: target depends on another field's value at runtime
                options_field = field.get("options", "")
                self.G.add_edge(
                    doctype,
                    "__dynamic__",
                    relation="dynamic_link",
                    fieldname=fieldname,
                    options_field=options_field,
                    label=field.get("label", fieldname),
                )

        elif fieldtype in CHILD_TABLE_TYPES:
            child = field.get("options", "")
            if child:
                if not self.G.has_node(child):
                    self.G.add_node(child, type="doctype")

                self.G.add_edge(
                    doctype,
                    child,
                    relation="child_table",
                    fieldname=fieldname,
                    label=field.get("label", fieldname),
                )
                self.stats["child_table_edges"] += 1

    def _process_hooks(self, app_path: Path) -> None:
        hooks_file = app_path / app_path.name / "hooks.py"
        if not hooks_file.exists():
            return

        try:
            source = hooks_file.read_text()
        except OSError:
            return

        # Extract doc_events block: doc_events = { "DocType": { "event": "handler" } }
        doc_events = self._extract_doc_events(source, app_path.name)
        for doctype, events in doc_events.items():
            if not self.G.has_node(doctype):
                self.G.add_node(doctype, type="doctype", app=app_path.name)
            for event, handlers in events.items():
                for handler in handlers:
                    self.G.add_edge(
                        doctype,
                        handler,
                        relation="hook",
                        event=event,
                        app=app_path.name,
                    )
                    self.stats["hook_rules"] += 1

        # Extract override_whitelisted_methods, fixtures, scheduler_events as node metadata
        self._extract_scheduler_events(source, app_path.name)

    def _extract_doc_events(self, source: str, app_name: str) -> dict[str, dict[str, list[str]]]:
        result: dict[str, dict[str, list[str]]] = {}

        # Find doc_events dict via regex — handles multiline Python dicts
        match = re.search(r"doc_events\s*=\s*\{", source)
        if not match:
            return result

        start = match.end() - 1  # position of opening {
        depth = 0
        i = start
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        block = source[start : i + 1]

        # Parse doctype entries
        doctype_pat = re.compile(r'"([^"]+)"\s*:\s*\{([^{}]+)\}', re.DOTALL)
        event_pat = re.compile(r'"([^"]+)"\s*:\s*["\[]([^"\]]+)["\]]')

        for dt_match in doctype_pat.finditer(block):
            doctype = dt_match.group(1)
            events_block = dt_match.group(2)
            result[doctype] = {}

            for ev_match in event_pat.finditer(events_block):
                event = ev_match.group(1)
                if event not in DOC_EVENT_HOOKS:
                    continue
                handler_raw = ev_match.group(2)
                # May be a single string or comma-separated list
                handlers = [h.strip().strip('"').strip("'") for h in handler_raw.split(",") if h.strip()]
                result[doctype][event] = handlers

        return result

    def _extract_scheduler_events(self, source: str, app_name: str) -> None:
        match = re.search(r"scheduler_events\s*=\s*\{", source)
        if not match:
            return

        start = match.end() - 1
        depth = 0
        i = start
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        block = source[start : i + 1]
        handler_pat = re.compile(r'"([a-z_.]+\.[a-z_.]+)"')
        handlers = handler_pat.findall(block)

        for handler in handlers:
            node_id = f"scheduler::{app_name}::{handler}"
            self.G.add_node(node_id, type="scheduler_job", app=app_name, handler=handler)

    def _print_stats(self) -> None:
        print("\n--- Graph Stats ---")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        print(f"  total nodes: {self.G.number_of_nodes()}")
        print(f"  total edges: {self.G.number_of_edges()}")
