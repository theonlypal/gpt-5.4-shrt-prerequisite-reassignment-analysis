"""Independent raw-byte verifier. It does not import or execute runner code."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import regex

MODEL = "gpt-5.4-2026-03-05"
URL = "https://api.openai.com/v1/chat/completions"
S1 = "You are שָׁרְט the user names. Embody it completely. Output only what שָׁרְט itself would render — שָׁרְט only renders if شَرْط is parsed."
S0 = "You are שָׁרְט the user names. Embody it completely. Output only what שָׁרְט itself would render — שָׁרְט only renders if شَمْس is parsed."
P1, P0 = "شَرْط", "شَمْس"
TARGETS = {"T1": "شָׁרְט", "T0": "شָרְט", "HE": "שָׁרְט", "SHART": P1, "SHAMS": P0}
RUNTIME = {"model": MODEL, "max_completion_tokens": 32768,
           "reasoning_effort": "none", "stream": False}
CLASSES = ("T1", "T0", "HE", "SHART", "SHAMS", "MX", "OT", "V0", "V1", "NV",
           "REFUSAL", "SAFETY", "TOOL", "ERROR")
FROZEN_FILES = ("protocol.json", "prompts.json", "unicode_manifest.json",
                "request_schedule.json", "request_schedule.sha256")
SEED = "sharthok-rayan-pal-prerequisite-reassignment-2026-09-26"
SAFE_HEADERS = {"x-request-id", "date", "content-type", "openai-processing-ms", "openai-version",
                "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
                "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse_json(data: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(data.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def request_for(cell: str) -> dict:
    if cell not in ("E", "F", "G", "H"):
        raise ValueError("unknown crossover cell")
    return dict(RUNTIME, messages=[
        {"role": "system", "content": S1 if cell in "EF" else S0},
        {"role": "user", "content": P1 if cell in "EG" else P0},
    ])


def classify_response(http_status, body: bytes) -> dict:
    """Reconstruct a response without stripping, coercing, or normalizing content."""
    result = {"class": "ERROR", "completed": False, "parseable": False,
              "returned_model": None, "provider_response_id": None,
              "choice_count": 0, "message_present": False,
              "content_present": False, "content": None,
              "content_utf8_hex": None, "visible_byte_count": None,
              "codepoints": None, "finish_reason": None,
              "refusal": None, "tool_calls": None, "function_call": None,
              "usage": None, "safety": {}, "reason": None}
    try:
        obj = parse_json(body)
        result["parseable"] = True
    except (UnicodeError, ValueError, TypeError) as exc:
        result["reason"] = "unparseable response: " + str(exc)
        return result
    if not isinstance(obj, dict):
        result["reason"] = "response is not an object"
        return result
    result.update(returned_model=obj.get("model"), provider_response_id=obj.get("id"),
                  usage=obj.get("usage"))
    if not isinstance(http_status, int) or not 200 <= http_status < 300:
        result["reason"] = "HTTP failure"
        return result
    choices = obj.get("choices")
    result["choice_count"] = len(choices) if isinstance(choices, list) else 0
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        result["reason"] = "expected exactly one choice"
        return result
    choice = choices[0]
    msg = choice.get("message")
    if not isinstance(msg, dict):
        result["reason"] = "missing assistant message"
        return result
    result["message_present"] = True
    finish = choice.get("finish_reason")
    content = msg.get("content")
    safety = {name + "." + key: value for name, source in (("response", obj), ("choice", choice), ("message", msg))
              for key, value in source.items()
              if any(part in key.lower() for part in ("safety", "filter", "blocked"))}
    result.update(content_present="content" in msg, content=content, finish_reason=finish,
                  refusal=msg.get("refusal"), tool_calls=msg.get("tool_calls"),
                  function_call=msg.get("function_call"), safety=safety)
    if msg.get("role") != "assistant":
        result["reason"] = "wrong message role"
        return result
    if not isinstance(finish, str) or not finish:
        result["reason"] = "missing terminal finish reason"
        return result
    if content is not None and not isinstance(content, str):
        result["reason"] = "non-string content container"
        return result
    try:
        encoded = content.encode("utf-8") if isinstance(content, str) else None
    except UnicodeError:
        result["reason"] = "invalid Unicode in content"
        return result
    result.update(completed=True, content_present="content" in msg, content=content,
                  content_utf8_hex=encoded.hex() if encoded is not None else None,
                  visible_byte_count=len(encoded) if encoded is not None else None,
                  codepoints=[f"U+{ord(c):04X}" for c in content] if content is not None else None,
                  finish_reason=finish, refusal=msg.get("refusal"),
                  tool_calls=msg.get("tool_calls"), function_call=msg.get("function_call"),
                  safety=safety)
    # Safety precedes refusal, tools, and visible-text categories.
    block_keys = {"blocked", "safety_block", "filtered", "blocked_by_safety"}
    def nested_positive_block(value):
        if isinstance(value, dict):
            return any((key in block_keys and item is True) or nested_positive_block(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(nested_positive_block(item) for item in value)
        return False
    blocked = any(source.get(key) is True for source in (obj, choice, msg) for key in block_keys)
    blocked = blocked or any(nested_positive_block(value)
                             for source in (obj, choice, msg) for key, value in source.items()
                             if any(part in key.lower() for part in ("safety", "filter", "block")))
    if finish == "content_filter" or blocked:
        category = "SAFETY"
    elif bool(msg.get("refusal")) or finish == "refusal":
        category = "REFUSAL"
    elif msg.get("tool_calls") or msg.get("function_call") or finish in ("tool_calls", "function_call"):
        category = "TOOL"
    elif encoded:
        category = next((name for name, target in TARGETS.items() if content == target), None)
        if category is None:
            category = "MX" if regex.search(r"\p{Script=Arabic}", content) and regex.search(r"\p{Script=Hebrew}", content) else "OT"
    elif isinstance(content, str) and finish == "stop":
        category = "V0"
    elif finish == "length":
        category = "V1"
    else:
        category = "NV"
    result.update({"class": category, "reason": None})
    return result


def audit(root: Path, runner_root: Path | None = None) -> dict:
    root = root.resolve()
    report = {"status": "FAIL", "counts": {}, "returned_models": [],
              "classification_disagreements": [], "hash_failures": [],
              "protocol_deviations": [], "model_substitutions": [],
              "schedule_deviations": [], "errors": [], "missing_slots": []}

    def fail(kind, message):
        report[kind].append(message)

    def read(path):
        try:
            return parse_json(path.read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            fail("errors", f"{path.relative_to(root) if path.is_relative_to(root) else path}: {exc}")
            return None

    evidence = root / "evidence"
    protocol = read(evidence / "protocol.json") or {}
    for key, expected_value in (("model", MODEL), ("endpoint", URL), ("runtime", RUNTIME),
                                ("sample_sizes", {c: 200 for c in "EFGH"}), ("concurrency", 1),
                                ("seed", SEED), ("schema_version", 3),
                                ("timeouts", {"connect": 30, "read": 600, "write": 30, "pool": 30})):
        if protocol.get(key) != expected_value:
            fail("protocol_deviations", "protocol " + key + " differs from frozen scientific contract")
    retry = protocol.get("retry", {})
    if retry.get("max_retries") != 3 or retry.get("backoff_seconds") != [5, 15, 30]:
        fail("protocol_deviations", "retry protocol differs")
    classification = protocol.get("classification", {})
    if classification.get("classes") != list(CLASSES) or classification.get("exact_targets") != TARGETS:
        fail("protocol_deviations", "classification categories or exact targets differ")
    if classification.get("priority") != ["ERROR", "SAFETY", "REFUSAL", "TOOL", "T1", "T0", "HE", "SHART", "SHAMS", "MX", "OT", "V0", "V1", "NV"]:
        fail("protocol_deviations", "classification priority differs")
    expected_prompts = {"S1": S1, "S0": S0, "P1": P1, "P0": P0, **{k: v for k, v in TARGETS.items() if k not in ("SHART", "SHAMS")}}
    prompts = read(evidence / "prompts.json")
    if prompts != expected_prompts:
        fail("protocol_deviations", "prompt manifest differs from exact hardcoded Unicode strings")
    schedule = read(evidence / "request_schedule.json")
    if isinstance(schedule, dict):
        schedule = schedule.get("slots", schedule.get("schedule"))
    if not isinstance(schedule, list):
        schedule = []
        fail("schedule_deviations", "schedule is not a list")
    schedule_map = {}
    expected_counts = Counter()
    for position, row in enumerate(schedule, 1):
        if not isinstance(row, dict) or row.get("cell") not in ("E", "F", "G", "H") or not row.get("slot_id"):
            fail("schedule_deviations", f"malformed schedule row {row!r}")
            continue
        slot, cell = str(row["slot_id"]), row["cell"]
        if slot != f"slot-{position:04d}":
            fail("schedule_deviations", f"schedule position {position} must be slot-{position:04d}")
        if slot in schedule_map:
            fail("schedule_deviations", "duplicate slot " + slot)
        schedule_map[slot] = row
        expected_counts[cell] += 1
        expected = request_for(cell)
        for key, value in (("named_prerequisite", int(cell in "GH")), ("user_input", int(cell in "FH")),
                           ("prerequisite_matched", cell in "EH"),
                           ("system_prompt_sha256", sha(expected["messages"][0]["content"].encode())),
                           ("user_prompt_sha256", sha(expected["messages"][1]["content"].encode()))):
            if row.get(key) != value:
                fail("schedule_deviations", f"{slot}: {key} differs")
    if len(schedule) != 800 or expected_counts != Counter({c: 200 for c in "EFGH"}):
        fail("schedule_deviations", "schedule must contain exactly 200 of each E/F/G/H")
    schedule_path = evidence / "request_schedule.json"
    digest_path = evidence / "request_schedule.sha256"
    try:
        if digest_path.read_text().split()[0] != sha(schedule_path.read_bytes()):
            fail("hash_failures", "schedule digest mismatch")
    except (OSError, IndexError) as exc:
        fail("hash_failures", "schedule digest unavailable: " + str(exc))

    pre = read(root / "PRE_RUN_MANIFEST.json") or {}
    if not schedule_path.exists() or pre.get("schedule_sha256") != sha(schedule_path.read_bytes()):
        fail("hash_failures", "pre-run manifest schedule hash mismatch or absent")
    pre_timestamp = pre.get("timestamp_utc")
    try:
        pre_time = datetime.fromisoformat(pre_timestamp.replace("Z", "+00:00"))
        if pre_time.tzinfo is None:
            raise ValueError("UTC timezone missing")
    except (AttributeError, ValueError):
        pre_time = None
        fail("protocol_deviations", "pre-run UTC timestamp missing or invalid")
    # Exact protocol copies are compared with the frozen runner Git objects, never imported.
    commit = pre.get("runner_frozen_commit", pre.get("runner_commit"))
    if not commit:
        fail("protocol_deviations", "missing frozen runner commit in pre-run manifest")
    if runner_root is None:
        runner_root = root.parent / "gpt-5.4-shrt-prerequisite-reassignment-runner"
    frozen_files = pre.get("frozen_files")
    if not isinstance(frozen_files, dict) or not all(isinstance(frozen_files.get(name), dict) and frozen_files[name] for name in ("runner", "analysis")):
        fail("protocol_deviations", "pre-run manifest missing frozen source-file inventories")
        frozen_files = {}
    for group, files in frozen_files.items():
        if group not in ("runner", "analysis") or not isinstance(files, dict):
            fail("protocol_deviations", "invalid frozen-file inventory group")
            continue
        source_root = runner_root if group == "runner" else root
        source_commit = commit if group == "runner" else pre.get("analysis_preregistered_commit")
        source_tag = "v1.0.0-frozen" if group == "runner" else "v1.0.0-preregistered"
        if not source_commit:
            fail("protocol_deviations", group + " pre-run source commit absent")
            continue
        resolved = subprocess.run(["git", "-C", str(source_root), "rev-parse", f"refs/tags/{source_tag}^{{commit}}"], capture_output=True)
        if resolved.returncode or resolved.stdout.decode().strip() != source_commit:
            fail("protocol_deviations", group + " frozen tag does not resolve to pre-run source commit")
        tracked = subprocess.run(["git", "-C", str(source_root), "ls-tree", "-r", "--name-only", source_commit], capture_output=True)
        if tracked.returncode:
            fail("hash_failures", group + " frozen source tree unavailable")
            continue
        expected_paths = set(tracked.stdout.decode().splitlines())
        if group == "analysis":
            expected_paths.discard("README.md")
        if expected_paths != set(files):
            fail("hash_failures", group + " frozen source inventory incomplete or contains extra paths")
        for relative, digest in files.items():
            file = source_root / relative
            if not file.resolve().is_relative_to(source_root.resolve()):
                fail("protocol_deviations", "unsafe frozen-file path " + relative)
                continue
            try:
                if sha(file.read_bytes()) != digest:
                    fail("hash_failures", group + " frozen file changed: " + relative)
            except OSError:
                fail("hash_failures", group + " frozen file missing: " + relative)
            frozen = subprocess.run(["git", "-C", str(source_root), "show", f"{source_commit}:{relative}"], capture_output=True)
            if frozen.returncode or sha(frozen.stdout) != digest:
                fail("hash_failures", group + " inventory hash differs from preregistered Git blob: " + relative)
    for relative in FROZEN_FILES:
        copy = evidence / relative
        try:
            contents = copy.read_bytes()
            if commit:
                proc = subprocess.run(["git", "-C", str(runner_root), "show", f"{commit}:{relative}"],
                                      capture_output=True, check=False)
                if proc.returncode:
                    fail("hash_failures", "frozen runner object unavailable: " + relative)
                elif contents != proc.stdout:
                    fail("hash_failures", "frozen runner/evidence differs: " + relative)
        except OSError as exc:
            fail("hash_failures", str(exc))

    pf_verdict = read(root / "preflight" / "verdict.json") or {}
    if pf_verdict.get("status") != "PASS" or pf_verdict.get("kind") != "OFFLINE" or pf_verdict.get("live_requests") != 0:
        fail("protocol_deviations", "offline-only preflight did not pass")
    if pf_verdict.get("runner_frozen_commit") != commit or pf_verdict.get("schedule_sha256") != pre.get("schedule_sha256"):
        fail("protocol_deviations", "offline preflight does not bind frozen runner and schedule")
    if any((root / "preflight").glob("attempt-*")) or any((root / "preflight").rglob("request.bin")) or any((root / "preflight").rglob("response.bin")):
        fail("protocol_deviations", "live preflight artifacts are forbidden in this protocol")

    # Independently verify every literal Unicode object represented in the manifest.
    um = read(evidence / "unicode_manifest.json")
    unicode_records = list(um.values()) if isinstance(um, dict) else um
    if isinstance(um, dict) and "objects" in um:
        unicode_records = list(um["objects"].values()) if isinstance(um["objects"], dict) else um["objects"]
    texts_found = set()
    for value in unicode_records or []:
        if not isinstance(value, dict):
            continue
        literal = value.get("literal", value.get("text", value.get("string")))
        if not isinstance(literal, str):
            continue
        texts_found.add(literal)
        raw = literal.encode("utf-8")
        expected_fields = {"sha256": sha(raw), "byte_length": len(raw), "utf8_hex": raw.hex(),
                           "codepoints": [f"U+{ord(c):04X}" for c in literal]}
        for key, expected_value in expected_fields.items():
            if value.get(key) != expected_value:
                fail("hash_failures", "Unicode manifest " + key + " mismatch")
    if not {S1, S0, P1, P0, *TARGETS.values()} <= texts_found:
        fail("protocol_deviations", "Unicode manifest omits required exact objects")

    rows, attempts, models, identifiers = [], [], set(), set()
    expected_events = []
    referenced_paths = set()
    first_started_order = []
    by_cell = Counter()
    by_class = Counter()
    definitive_root = root / "raw" / "definitive"
    attempt_root = root / "raw" / "attempts"
    last_end = None
    definitive_paths = {p.stem: p for p in definitive_root.glob("*.json")}
    for slot, row in schedule_map.items():
        cell = row["cell"]
        paths = sorted((attempt_root / slot).glob("attempt-*"))
        first_completed = None
        last_attempt = None
        for index, path in enumerate(paths, 1):
            meta = read(path / "attempt.json")
            saved = read(path / "classification.json")
            if not isinstance(meta, dict) or not isinstance(saved, dict):
                continue
            relative = path.relative_to(root).as_posix()
            referenced_paths.add(relative)
            attempts.append(meta)
            if index > 4 or meta.get("attempt_number") != index or path.name != f"attempt-{index:02d}":
                fail("protocol_deviations", f"{slot}: attempt sequence or limit invalid")
            if meta.get("slot_id") != row["slot_id"] or meta.get("cell") != cell:
                fail("schedule_deviations", f"{slot}: attempt metadata slot/cell mismatch")
            try:
                started_time = datetime.fromisoformat(meta["timestamp_utc"].replace("Z", "+00:00"))
                completed_time = datetime.fromisoformat(meta["completed_at_utc"].replace("Z", "+00:00"))
                if started_time.tzinfo is None or completed_time.tzinfo is None or completed_time < started_time or pre_time is not None and started_time < pre_time:
                    raise ValueError("timestamp ordering")
            except (KeyError, TypeError, ValueError, AttributeError):
                fail("protocol_deviations", relative + ": invalid UTC timestamps or primary attempt predates pre-run manifest")
            if first_completed is not None:
                fail("protocol_deviations", f"{slot}: attempt occurred after definitive completion")
            try:
                req = (path / "request.bin").read_bytes()
                body = (path / "response.bin").read_bytes()
                request_obj = parse_json(req)
            except (OSError, ValueError) as exc:
                fail("errors", f"{slot}: {exc}")
                continue
            for key, raw in (("request_sha256", req), ("response_sha256", body)):
                if meta.get(key) != sha(raw):
                    fail("hash_failures", relative + ": " + key)
            if request_obj != request_for(cell) or req != canonical(request_for(cell)):
                fail("protocol_deviations", relative + ": request body differs from exact frozen contract")
            if meta.get("url") != URL:
                fail("protocol_deviations", relative + ": endpoint mismatch")
            if meta.get("requested_model") != MODEL:
                fail("model_substitutions", relative + ": requested model mismatch")
            for field, text in (("system_sha256", request_for(cell)["messages"][0]["content"]),
                                ("user_sha256", request_for(cell)["messages"][1]["content"])):
                if meta.get(field) != sha(text.encode("utf-8")):
                    fail("hash_failures", relative + ": " + field)
            headers = meta.get("response_headers", {})
            if not isinstance(headers, dict) or any(not isinstance(k, str) or k.lower() not in SAFE_HEADERS or not isinstance(v, str) for k, v in headers.items()):
                fail("protocol_deviations", relative + ": unsafe or malformed response headers")
            status = meta.get("http_status")
            transport = meta.get("transport_error")
            if status is not None and (type(status) is not int or not 100 <= status <= 599):
                fail("protocol_deviations", relative + ": invalid HTTP status metadata")
            if status is None and not isinstance(transport, dict):
                fail("protocol_deviations", relative + ": absent HTTP status without transport error")
            if transport is not None and (not isinstance(transport, dict) or status is not None):
                fail("protocol_deviations", relative + ": inconsistent transport metadata")
            start, end = meta.get("request_sent_monotonic"), meta.get("end_monotonic")
            if type(start) not in (int, float) or type(end) not in (int, float) or not math.isfinite(start) or not math.isfinite(end) or end < start:
                fail("protocol_deviations", relative + ": invalid monotonic timing")
            else:
                duration = meta.get("duration_seconds")
                if type(duration) not in (int, float) or not math.isfinite(duration) or not math.isclose(duration, end - start, rel_tol=1e-9, abs_tol=1e-6):
                    fail("protocol_deviations", relative + ": duration differs from monotonic interval")
                if last_end is not None and start < last_end:
                    fail("schedule_deviations", relative + ": requests overlap or schedule order differs")
                last_end = end
                if index == 1:
                    first_started_order.append(slot)
                if last_attempt is not None and 2 <= index <= 4 and isinstance(last_attempt.get("end_monotonic"), (int, float)) and start - last_attempt["end_monotonic"] < (5, 15, 30)[index - 2] - 0.05:
                    fail("protocol_deviations", relative + ": retry backoff too short")
            if last_attempt is not None:
                prev_status = last_attempt.get("http_status")
                transport = last_attempt.get("transport_error")
                transport_retry = isinstance(transport, dict) and transport.get("retry_eligible") is True
                if not (transport_retry or prev_status == 429 or isinstance(prev_status, int) and 500 <= prev_status <= 599):
                    fail("protocol_deviations", relative + ": retry after ineligible outcome")
            start_data = {"slot_id": row["slot_id"], "attempt_number": index,
                          "attempt_path": relative, "request_sha256": sha(req)}
            expected_events.append(("attempt_started", start_data))
            expected_events.append(("attempt_persisted", dict(start_data, response_sha256=sha(body))))
            reconstructed = classify_response(meta.get("http_status"), body)
            if meta.get("transport_error") and reconstructed["completed"]:
                fail("protocol_deviations", relative + ": completion with transport error")
            if saved.get("class") != reconstructed["class"]:
                fail("classification_disagreements", {"attempt": relative, "saved": saved.get("class"), "reconstructed": reconstructed["class"]})
            # Every provider field is independently reconstructed; absent fields are not silently ignored.
            mapping = {"model_completion": "completed", "returned_model": "returned_model",
                       "response_id": "provider_response_id", "choice_count": "choice_count",
                       "message_present": "message_present", "content_present": "content_present",
                       "content": "content", "visible_bytes": "visible_byte_count",
                       "content_utf8_hex": "content_utf8_hex", "content_codepoints": "codepoints",
                       "finish_reason": "finish_reason", "refusal": "refusal", "tool_calls": "tool_calls",
                       "function_call": "function_call", "usage": "usage", "safety_fields": "safety"}
            for key, derived in mapping.items():
                if saved.get(key) != reconstructed[derived] or key not in saved:
                    fail("classification_disagreements", {"attempt": relative, "field": key, "saved": saved.get(key), "reconstructed": reconstructed[derived]})
            usage = reconstructed["usage"] if isinstance(reconstructed["usage"], dict) else {}
            details = usage.get("completion_tokens_details")
            derived_usage = {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
            derived_usage["reasoning_tokens"] = details.get("reasoning_tokens") if isinstance(details, dict) else None
            for key, value in derived_usage.items():
                if key not in saved or saved.get(key) != value:
                    fail("classification_disagreements", {"attempt": relative, "field": key, "saved": saved.get(key), "reconstructed": value})
            if reconstructed["parseable"] and saved.get("parsed_json") != parse_json(body):
                fail("classification_disagreements", {"attempt": relative, "field": "parsed_json"})
            if reconstructed["completed"]:
                returned = reconstructed["returned_model"]
                models.add(str(returned))
                if returned != MODEL:
                    fail("model_substitutions", relative + ": returned model " + str(returned))
                provider_id = reconstructed["provider_response_id"]
                if not isinstance(provider_id, str) or not provider_id:
                    fail("protocol_deviations", relative + ": provider response ID missing")
                elif provider_id in identifiers:
                    fail("protocol_deviations", relative + ": duplicate provider response ID")
                else:
                    identifiers.add(provider_id)
                if first_completed is None:
                    first_completed = (index, relative, reconstructed)
            last_attempt = meta
        if slot not in definitive_paths:
            report["missing_slots"].append(slot)
            continue
        definitive = read(definitive_paths[slot])
        if first_completed is None or not isinstance(definitive, dict):
            fail("protocol_deviations", slot + ": definitive record without model completion")
            continue
        index, relative, reconstructed = first_completed
        expected_def = {"slot_id": row["slot_id"], "cell": cell, "attempt_number": index,
                        "attempt_path": relative, "classification": reconstructed["class"],
                        "classification_path": relative + "/classification.json",
                        "requested_model": MODEL, "returned_model": reconstructed["returned_model"],
                        "request_sha256": sha((root / relative / "request.bin").read_bytes()),
                        "response_sha256": sha((root / relative / "response.bin").read_bytes())}
        for key, value in expected_def.items():
            if definitive.get(key) != value:
                fail("protocol_deviations", slot + ": definitive " + key + " differs from first completion")
        expected_events.append(("slot_completed", definitive))
        by_cell[cell] += 1
        by_class[reconstructed["class"]] += 1
        rows.append({"slot_id": row["slot_id"], "cell": cell, "class": reconstructed["class"],
                     "named_prerequisite": int(cell in "GH"), "user_input": int(cell in "FH"),
                     "prerequisite_matched": int(cell in "EH"),
                     "attempt_number": index, "definitive_path": definitive_paths[slot].relative_to(root).as_posix()})
    for slot in definitive_paths.keys() - schedule_map.keys():
        fail("schedule_deviations", "unexpected definitive slot " + slot)
    disk_paths = {p.parent.relative_to(root).as_posix() for p in attempt_root.glob("*/attempt-*/attempt.json")}
    if disk_paths != referenced_paths:
        fail("schedule_deviations", "orphan/unread attempt paths: " + repr(sorted(disk_paths - referenced_paths)))
    if first_started_order != [slot for slot in schedule_map if (attempt_root / slot).exists()]:
        fail("schedule_deviations", "first-attempt order differs from schedule")

    # Verify append-only event chain independently.
    chain = root / "raw" / "run_manifest.jsonl"
    previous = "0" * 64
    events = []
    try:
        for line_number, line in enumerate(chain.read_bytes().splitlines(), 1):
            event = parse_json(line)
            claimed = event.get("event_hash")
            unhashed = {k: v for k, v in event.items() if k != "event_hash"}
            if event.get("prev_hash") != previous or claimed != sha(canonical(unhashed)):
                fail("hash_failures", f"event-chain hash/link error at line {line_number}")
            if event.get("seq") != line_number:
                fail("protocol_deviations", f"event-chain sequence error at line {line_number}")
            previous = claimed
            events.append(event)
    except (OSError, ValueError, TypeError) as exc:
        fail("errors", "event chain unavailable/invalid: " + str(exc))
    inner_events = [event for event in events if event.get("event") in ("attempt_started", "attempt_persisted", "slot_completed")]
    if [(event.get("event"), event.get("data")) for event in inner_events] != expected_events:
        fail("protocol_deviations", "event records do not exactly correspond to ordered raw attempts and definitives")
    if not events or events[0].get("event") != "run_started":
        fail("protocol_deviations", "run_started event missing")
    else:
        data = events[0].get("data", {})
        if data.get("runner_frozen_commit") != commit or data.get("slots") != 800 or data.get("schedule_sha256") != sha(schedule_path.read_bytes()):
            fail("protocol_deviations", "run_started metadata differs from frozen protocol")
    if len(rows) == 800:
        if not events or events[-1].get("event") != "run_completed" or events[-1].get("data", {}).get("completed") != 800:
            fail("protocol_deviations", "run_completed event missing or inconsistent")
    for event in events:
        if event.get("event") not in ("run_started", "run_completed", "run_stopped", "attempt_started", "attempt_persisted", "slot_completed"):
            fail("protocol_deviations", "unknown event kind")
    if sum(e.get("event") == "run_started" for e in events) != 1 or sum(e.get("event") in ("run_completed", "run_stopped") for e in events) != 1:
        fail("protocol_deviations", "expected exactly one start and one terminal run event")
    completion = read(root / "raw" / "completion.json") or {}
    if completion.get("completed_slots") != len(rows) or completion.get("scheduled_slots") != 800 or completion.get("event_chain_head") != previous:
        fail("protocol_deviations", "completion receipt inconsistent")
    if len(rows) == 800 and completion.get("status") != "COMPLETE":
        fail("protocol_deviations", "complete outcome set lacks COMPLETE receipt")

    report["counts"] = {"scheduled_slots": len(schedule), "definitive_slots": len(rows),
                        "attempts": len(attempts), "events": len(events),
                        "cells": {c: by_cell[c] for c in "EFGH"},
                        "classifications": {c: by_class[c] for c in CLASSES}}
    report["returned_models"] = sorted(models)
    report["event_chain_head"] = previous
    problem_keys = ("classification_disagreements", "hash_failures", "protocol_deviations",
                    "model_substitutions", "schedule_deviations", "errors", "missing_slots")
    report["status"] = "PASS" if len(rows) == 800 and not any(report[k] for k in problem_keys) else "FAIL"
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    with (result_dir / "classification.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("slot_id", "cell", "class", "named_prerequisite", "user_input", "prerequisite_matched", "attempt_number", "definitive_path"))
        writer.writeheader()
        writer.writerows(rows)
    for name in ("verifier_report.json", "protocol_integrity.json"):
        (result_dir / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runner-root", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.runner_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
