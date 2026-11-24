"""Run the two offline EGE-UNet evidence-collection conditions."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import time

from validate import ROOT, citation_entries, sha256, validate

PACK = ROOT / "source-pack"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def save_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def short_run_evidence(metrics):
    splits = metrics["splits"]
    test = metrics["test"]
    paper = metrics["paper_reference"]
    return {
        "L1": f"使用 ISIC2017 官方划分，训练集抽取 {splits['train']:,} 张，保留 "
              f"{splits['val']} 张验证图和 {splits['test']} 张测试图；训练 {metrics['epochs']} 轮，"
              f"以第 {metrics['best_epoch']} 轮的权重测试，二值化阈值为 {metrics['threshold']}。",
        "L2": f"测试 Dice 为 {test['dice']:.2%}，前景 IoU 为 {test['iou']:.2%}，"
              f"模型共有 {metrics['parameters']:,} 个参数。",
        "L3": f"与原论文表 1 的均值相比，Dice 低 {(paper['dice']-test['dice'])*100:.2f} 个百分点，"
              f"前景 IoU 低 {(paper['iou']-test['iou'])*100:.2f} 个百分点。",
    }


class SourceTools:
    def __init__(self, output):
        self.catalog = {entry["id"]: entry for entry in load_json(PACK / "catalog.json")}
        self.output = output
        self.found = set()
        self.reads = {}

    def search_catalog(self, query):
        terms = set(query.casefold().split())
        matched = []
        for source_id, entry in self.catalog.items():
            score = sum(word.casefold() in terms for word in entry["keywords"])
            if score:
                matched.append({"source_id": source_id, "title": entry["title"], "score": score})
        matched.sort(key=lambda item: (-item["score"], item["source_id"]))
        self.found.update(item["source_id"] for item in matched)
        return matched

    def read_source(self, source_id):
        if source_id not in self.found:
            raise ValueError("source_id must come from search_catalog")
        entry = self.catalog[source_id]
        path = (PACK / entry["file"]).resolve()
        if path.parent != PACK:
            raise ValueError("source path must stay inside source-pack")
        content = load_json(path)
        evidence = short_run_evidence(content) if entry["role"] == "reproduction" else content
        result = {"source_id": source_id, "title": entry["title"], "role": entry["role"],
                  "origin": entry["origin"], "file": f"source-pack/{path.name}",
                  "sha256": sha256(path), "evidence": evidence}
        self.reads[source_id] = result
        return result

    def write_brief(self, title, sections, citations):
        draft = f"# {title}\n"
        for heading, text in sections:
            draft += f"\n## {heading}\n\n{text}\n"
        entries = citation_entries(draft)
        if set(citations) != {source_id for source_id, _, _ in entries} or len(entries) != 3:
            raise ValueError("brief needs one citation for each of the three sources")
        for source_id, first, last in entries:
            if source_id not in self.reads or not (1 <= first <= last) or any(
                f"L{n}" not in self.reads[source_id]["evidence"] for n in range(first, last + 1)
            ):
                raise ValueError("citation must refer to evidence actually read")
        path = self.output / "final-brief.md"
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(draft)
        return {"path": path.name, "sha256": sha256(path)}


def action_for(observations, contract):
    if not observations:
        return "search_catalog", {"query": contract["query"]}
    search = observations[0]["result"]
    required_count = contract["relevant_sources_to_read"]
    reads = [item["result"] for item in observations if item["tool"] == "read_source"]
    if len(reads) < required_count:
        if len(search) < required_count:
            raise ValueError("search returned fewer than three sources")
        return "read_source", {"source_id": search[len(reads)]["source_id"]}
    by_role = {entry["role"]: entry for entry in reads}
    paper, review, reproduction = (by_role[role] for role in ("paper", "review", "reproduction"))
    p, r, m = (entry["evidence"] for entry in (paper, review, reproduction))
    sections = [
        ("方法与成本", f"{p['L1']}{p['L2']} [{paper['source_id']}:L1-L2]"),
        ("文献对照", f"{r['L1']} [{review['source_id']}:L1]"),
        ("短训结果", f"{m['L1']}{m['L2']}{m['L3']} [{reproduction['source_id']}:L1-L3]"),
        ("比较口径", "短训使用更少的训练图像和轮次，数据划分也与原论文不同。"
         "这个差距反映两套设置下的结果，现有记录无法将差距单独归因于训练轮次。"),
    ]
    return "write_brief", {"title": "EGE-UNet 低算力复现简报", "sections": sections,
                           "citations": [paper["source_id"], review["source_id"], reproduction["source_id"]]}


def snapshot(observations):
    return {
        "catalog_hits": len(observations[0]["result"]) if observations else 0,
        "sources_read": [item["result"]["source_id"] for item in observations
                         if item["tool"] == "read_source"],
        "brief_written": any(item["tool"] == "write_brief" for item in observations),
    }


def run_group(root, budget, contract):
    variant = f"max_steps_{budget}"
    folder = root / variant
    folder.mkdir()
    tools = SourceTools(folder)
    handlers = {"search_catalog": tools.search_catalog, "read_source": tools.read_source,
                "write_brief": tools.write_brief}
    trail, results = [], []
    start = time.perf_counter()
    for step in range(1, budget + 1):
        before = snapshot(results)
        tool, arguments = action_for(results, contract)
        observation = handlers[tool](**arguments)
        common = {"variant": variant, "step": step, "tool": tool, "arguments": arguments}
        results.append({**common, "result": observation})
        trail.append({**common, "before": before, "after": snapshot(results),
                      "observation_ref": f"{variant}/tool-results.jsonl:{step}"})
        if tool == "write_brief":
            break
    save_jsonl(folder / "trajectory.jsonl", trail)
    save_jsonl(folder / "tool-results.jsonl", results)
    status = validate(folder, variant, budget, contract)
    status["stop_reason"] = "brief_written" if snapshot(results)["brief_written"] else "step_budget_exhausted"
    status["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
    save_json(folder / "validation.json", status)
    return status, trail, results


def write_comparison(output, groups):
    columns = ["variant", "max_steps", "steps", "sources_read", "citations", "validator", "stop_reason"]
    save_csv(output / "comparison.csv", columns,
             [{column: status[column] for column in columns} for status in groups])
    pointers = {
        "brief_exists": "final-brief.md; write_brief in tool-results.jsonl",
        "three_unique_citations": "final-brief.md",
        "citations_were_read": "read_source in tool-results.jsonl; final-brief.md",
        "three_sources_read": "read_source in tool-results.jsonl",
        "within_step_budget": "trajectory.jsonl; validation.json",
        "brief_format": "final-brief.md; task-contract.json",
    }
    rows = []
    for status in groups:
        checks = {**status["gates"], "brief_format": status["brief_format"]}
        for check, passed in checks.items():
            rows.append({"variant": status["variant"], "check": check,
                         "result": "PASS" if passed else "FAIL",
                         "evidence": f"{status['variant']}/" + pointers[check]})
    save_csv(output / "evaluation.csv", ["variant", "check", "result", "evidence"], rows)


def write_provenance(output):
    root_inputs = [ROOT / name for name in (
        "run_agent.py", "validate.py", "task-contract.json", "tool-schema.json"
    )]
    root_inputs += sorted(PACK.glob("*.json"))
    outputs = sorted(path for path in output.rglob("*") if path.is_file())
    save_json(output / "provenance.json", {
        "mode": "local_rule_based_fixture",
        "fixture_id": "at3-ege-unet-source-pack-v1",
        "fixture_origin": "Student adaptation of W11 worksheet, with AT3 parts 1 and 2",
        "validator_origin": "Local implementation of the five gates listed in W11, section I",
        "api_used": False, "model_used": False, "network_used": False,
        "gpu_used": False, "third_party_framework_used": False,
        "environment": {"python": platform.python_version(), "os": platform.system()},
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in root_inputs},
        "outputs_sha256": {path.relative_to(output).as_posix(): sha256(path) for path in outputs},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("rerun"))
    args = parser.parse_args()
    output = args.output_dir
    if output.exists():
        parser.error("choose a new output directory to avoid using stale results")
    contract = load_json(ROOT / "task-contract.json")
    output.mkdir(parents=True)
    for name in ("task-contract.json", "tool-schema.json"):
        shutil.copyfile(ROOT / name, output / name)

    groups, trace, results = [], [], []
    budgets = contract["comparison"]["values"]
    for budget in budgets:
        status, group_trace, group_results = run_group(output, budget, contract)
        groups.append(status)
        trace.extend(group_trace)
        results.extend(group_results)
        print(f"{status['variant']}: {status['validator']}; "
              f"steps={status['steps']}; sources={status['sources_read']}; "
              f"citations={status['citations']}")
    save_jsonl(output / "trajectory.jsonl", trace)
    save_jsonl(output / "tool-results.jsonl", results)
    write_comparison(output, groups)

    passing = output / f"max_steps_{contract['max_steps']}" / contract["allowed_output"]
    if passing.is_file():
        shutil.copyfile(passing, output / contract["allowed_output"])
    error_note = ROOT / "error-note.md"
    if error_note.is_file():
        shutil.copyfile(error_note, output / error_note.name)
    write_provenance(output)
    if [group["validator"] for group in groups] != ["FAIL", "PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
