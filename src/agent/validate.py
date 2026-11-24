"""Check the final artifact against the observed tool results and step limit."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CITATION = re.compile(r"\[(S\d{2}):L(\d+)(?:-L(\d+))?\]")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def citation_entries(markdown):
    return [(source_id, int(first), int(last or first))
            for source_id, first, last in CITATION.findall(markdown)]


def validate(folder, variant, budget, contract):
    trajectory = read_jsonl(folder / "trajectory.jsonl")
    results = read_jsonl(folder / "tool-results.jsonl")
    written = [item["result"] for item in results if item["tool"] == "write_brief"]
    reads = {item["result"]["source_id"]: item["result"]["evidence"]
             for item in results if item["tool"] == "read_source"}
    brief = folder / contract["allowed_output"]
    contents = brief.read_text(encoding="utf-8") if brief.is_file() else ""
    entries = set(citation_entries(contents))
    sources = {entry[0] for entry in entries}
    citation_read = bool(entries) and all(
        1 <= first <= last <= len(reads.get(source_id, {}))
        and all(f"L{n}" in reads[source_id] for n in range(first, last + 1))
        for source_id, first, last in entries
    )
    log_keys = [(item["variant"], item["step"], item["tool"]) for item in trajectory]
    result_keys = [(item["variant"], item["step"], item["tool"]) for item in results]
    steps = len(trajectory)
    gates = {
        "brief_exists": bool(contents) and len(written) == 1
        and written[0]["sha256"] == sha256(brief),
        "three_unique_citations": len(entries) == len(sources) == 3
        and sources == set(contract["required_sources"]),
        "citations_were_read": citation_read,
        "three_sources_read": len(reads) == 3 and set(reads) == set(contract["required_sources"]),
        "within_step_budget": log_keys == result_keys and steps <= budget and
        [item[:2] for item in log_keys] == [(variant, n) for n in range(1, steps + 1)],
    }
    # A character cap keeps the Markdown brief short enough for a single page.
    sections = [line[3:] for line in contents.splitlines() if line.startswith("## ")]
    brief_format = bool(contents) and len(contents) <= contract["brief"]["max_chars"] and (
        sections == contract["brief"]["required_sections"]
    )
    return {
        "variant": variant, "max_steps": budget, "steps": steps,
        "sources_read": len(reads), "citations": len(entries), "gates": gates,
        "brief_format": brief_format, "brief_chars": len(contents),
        "validator": "PASS" if all(gates.values()) and brief_format else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run folder, e.g. evidence/max_steps_6")
    args = parser.parse_args()
    contract = json.loads((ROOT / "task-contract.json").read_text(encoding="utf-8"))
    variant = args.run.resolve().name
    if variant not in {"max_steps_3", "max_steps_6"}:
        parser.error("run folder must be named max_steps_3 or max_steps_6")
    result = validate(args.run, variant, int(variant[-1]), contract)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["validator"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
