"""Deterministic evidence indexes, hash manifest, and lossless results bundle."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

BUNDLE = "gpt-5.4-shrt-prerequisite-reassignment-results.zip"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, "Refusing to replace different evidence: " + str(path)
    else:
        with path.open("xb") as stream:
            stream.write(data)


def build(root):
    root = Path(root).resolve()
    with (root / "results/classification.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: r["slot_id"])
    by_id = {r["slot_id"]: r for r in rows}
    schedule = json.loads((root / "evidence/request_schedule.json").read_bytes())
    positions = {r["slot_id"]: i for i, r in enumerate(schedule)}

    def receipt(row):
        record_path = "raw/definitive/" + row["slot_id"] + ".json"
        record = json.loads((root / record_path).read_bytes())
        response_path = record["attempt_path"] + "/response.bin"
        request_path = record["attempt_path"] + "/request.bin"
        return {"slot_id": row["slot_id"], "cell": row["cell"], "class": row["class"],
                "definitive_path": record_path, "request_path": request_path,
                "request_sha256": sha((root / request_path).read_bytes()),
                "response_path": response_path, "response_sha256": sha((root / response_path).read_bytes())}

    for label, directory in (("T1", "exact_t1_examples"), ("V0", "v0_examples")):
        chosen = [r for r in rows if r["class"] == label][:3]
        write_json(root / "evidence" / directory / "index.json", {
            "selection": "First three occurrences in frozen schedule order; full corpus is the evidence",
            "class": label, "examples": [receipt(r) for r in chosen]})
    witnesses = {}
    for cell in "EFGH":
        selected = [row for row in rows if row["cell"] == cell][:1]
        witnesses[cell] = [receipt(row) for row in selected]
    write_json(root / "evidence/witness_cells/index.json", {
        "selection": "First occurrence of each cell in frozen schedule order; outcomes are not used for selection",
        "cells": witnesses})
    excluded = {"evidence/hash_manifest.json", BUNDLE, BUNDLE + ".sha256"}
    paths = sorted(p for p in root.rglob("*") if p.is_file()
                   and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts)
                   and not p.name.endswith((".pyc", ".zip"))
                   and p.relative_to(root).as_posix() not in excluded)
    manifest = {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in paths}
    write_json(root / "evidence/hash_manifest.json", {"algorithm": "sha256", "files": manifest,
        "exclusions": "This manifest, bundle/checksum, Git internals, hidden placeholders/caches, compiled Python"})
    paths.append(root / "evidence/hash_manifest.json")
    destination = root / BUNDLE
    assert not destination.exists(), "Bundle already exists"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(paths):
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    checksum = sha(destination.read_bytes())
    with (root / (BUNDLE + ".sha256")).open("x") as stream:
        stream.write(checksum + "  " + BUNDLE + "\n")
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name, expected in manifest.items():
            assert sha(archive.read(name)) == expected
    print(json.dumps({"bundle": str(destination), "sha256": checksum, "files": len(paths)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    build(parser.parse_args().root)
