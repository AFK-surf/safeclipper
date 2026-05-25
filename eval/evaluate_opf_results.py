#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from opf._api import OPF, RedactionResult


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def value_found(expected: str, haystack: str) -> bool:
    expected_n = normalize(expected)
    haystack_n = normalize(haystack)
    if not expected_n:
        return False
    if expected_n in haystack_n:
        return True
    if len(expected_n) >= 12:
        return expected_n[: max(8, int(len(expected_n) * 0.75))] in haystack_n
    return False


def detected(expected: str, spans: list[dict[str, object]]) -> tuple[bool, list[str]]:
    labels: list[str] = []
    for span in spans:
        span_text = str(span.get("text", ""))
        if value_found(expected, span_text) or value_found(span_text, expected):
            labels.append(str(span.get("label", "")))
    return bool(labels), labels


def expected_values_from_gallery(path: Path) -> dict[str, list[dict[str, str]]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    expected: dict[str, list[dict[str, str]]] = {}
    article_pattern = re.compile(r"<article\b.*?</article>", re.DOTALL | re.IGNORECASE)
    id_pattern = re.compile(r"<strong>(synthetic-pii(?:-extra)?-\d+)</strong>", re.IGNORECASE)
    value_pattern = re.compile(
        r"<li><code>(?P<category>.*?)</code>\s*<span>(?P<subtype>.*?)</span>:\s*"
        r"<strong>(?P<text>.*?)</strong></li>",
        re.DOTALL | re.IGNORECASE,
    )
    tag_pattern = re.compile(r"<[^>]+>")

    for article_match in article_pattern.finditer(source):
        article = article_match.group(0)
        id_match = id_pattern.search(article)
        if not id_match:
            continue
        scenario_id = html.unescape(id_match.group(1))
        values = []
        for value_match in value_pattern.finditer(article):
            values.append(
                {
                    "category": html.unescape(tag_pattern.sub("", value_match.group("category"))).strip(),
                    "subtype": html.unescape(tag_pattern.sub("", value_match.group("subtype"))).strip(),
                    "text": html.unescape(tag_pattern.sub("", value_match.group("text"))).strip(),
                }
            )
        if values:
            expected[scenario_id] = values
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ocr_jsonl")
    parser.add_argument("output_json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gallery-html", default=None)
    args = parser.parse_args()

    ocr_path = Path(args.ocr_jsonl)
    records = [json.loads(line) for line in ocr_path.read_text().splitlines() if line.strip()]
    gallery_expected = (
        expected_values_from_gallery(Path(args.gallery_html)) if args.gallery_html else {}
    )

    redactor = OPF(device=args.device, output_text_only=False)
    summary = {
        "input_records": len(records),
        "expected_source": str(args.gallery_html) if args.gallery_html else "ocr_jsonl",
        "values_total": 0,
        "ocr_value_hits": 0,
        "opf_value_hits": 0,
        "privacy_values_total": 0,
        "privacy_ocr_hits": 0,
        "privacy_opf_hits": 0,
        "utility_values_total": 0,
        "utility_ocr_hits": 0,
        "utility_opf_hits": 0,
        "records_with_any_expected_detected": 0,
        "records_with_all_ocr_visible_values_detected": 0,
        "values_by_category": Counter(),
        "ocr_hits_by_category": Counter(),
        "opf_hits_by_category": Counter(),
        "opf_labels": Counter(),
        "failures": [],
        "records": [],
    }

    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        text = record["text"]
        result = redactor.redact(text)
        if not isinstance(result, RedactionResult):
            raise TypeError("expected structured RedactionResult")
        spans = result.to_dict()["detected_spans"]
        for span in spans:
            summary["opf_labels"][span["label"]] += 1

        values = gallery_expected.get(record["scenarioID"], record["expectedValues"])
        record_values = []
        any_detected = False
        visible_detectable = 0
        visible_detected = 0
        for value in values:
            category = value["category"]
            expected = value["text"]
            summary["values_total"] += 1
            summary["values_by_category"][category] += 1
            if category == "privacy":
                summary["privacy_values_total"] += 1
            if category == "utility":
                summary["utility_values_total"] += 1

            ocr_hit = value_found(expected, text)
            opf_hit, labels = detected(expected, spans)
            if ocr_hit:
                summary["ocr_value_hits"] += 1
                summary["ocr_hits_by_category"][category] += 1
                if category == "privacy":
                    summary["privacy_ocr_hits"] += 1
                if category == "utility":
                    summary["utility_ocr_hits"] += 1
                visible_detectable += 1
            if opf_hit:
                summary["opf_value_hits"] += 1
                summary["opf_hits_by_category"][category] += 1
                if category == "privacy":
                    summary["privacy_opf_hits"] += 1
                if category == "utility":
                    summary["utility_opf_hits"] += 1
                any_detected = True
                if ocr_hit:
                    visible_detected += 1

            value_record = {
                "category": category,
                "subtype": value["subtype"],
                "expected": expected,
                "ocr_hit": ocr_hit,
                "opf_hit": opf_hit,
                "opf_labels": labels,
            }
            record_values.append(value_record)
            if not ocr_hit or not opf_hit:
                summary["failures"].append(
                    {
                        "scenario_id": record["scenarioID"],
                        "image_path": record["imagePath"],
                        **value_record,
                    }
                )

        if any_detected:
            summary["records_with_any_expected_detected"] += 1
        if visible_detectable and visible_detected == visible_detectable:
            summary["records_with_all_ocr_visible_values_detected"] += 1

        summary["records"].append(
            {
                "scenario_id": record["scenarioID"],
                "image_path": record["imagePath"],
                "ocr_chars": len(text),
                "ocr_tokens": len(record["tokens"]),
                "detected_span_count": len(spans),
                "values": record_values,
            }
        )
        if index % 10 == 0 or index == len(records):
            elapsed = time.perf_counter() - started
            print(f"OPF {index}/{len(records)} elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)

    elapsed = time.perf_counter() - started
    serializable = dict(summary)
    for key in [
        "values_by_category",
        "ocr_hits_by_category",
        "opf_hits_by_category",
        "opf_labels",
    ]:
        serializable[key] = dict(summary[key])
    serializable["elapsed_seconds"] = elapsed
    serializable["ocr_value_recall"] = (
        summary["ocr_value_hits"] / summary["values_total"] if summary["values_total"] else 0
    )
    serializable["end_to_end_value_recall"] = (
        summary["opf_value_hits"] / summary["values_total"] if summary["values_total"] else 0
    )
    serializable["opf_given_ocr_value_recall"] = (
        summary["opf_value_hits"] / summary["ocr_value_hits"] if summary["ocr_value_hits"] else 0
    )
    serializable["privacy_ocr_recall"] = (
        summary["privacy_ocr_hits"] / summary["privacy_values_total"]
        if summary["privacy_values_total"]
        else 0
    )
    serializable["privacy_end_to_end_recall"] = (
        summary["privacy_opf_hits"] / summary["privacy_values_total"]
        if summary["privacy_values_total"]
        else 0
    )
    serializable["privacy_opf_given_ocr_recall"] = (
        summary["privacy_opf_hits"] / summary["privacy_ocr_hits"]
        if summary["privacy_ocr_hits"]
        else 0
    )
    serializable["utility_false_positive_rate"] = (
        summary["utility_opf_hits"] / summary["utility_values_total"]
        if summary["utility_values_total"]
        else 0
    )

    Path(args.output_json).write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(json.dumps({k: serializable[k] for k in [
        "input_records",
        "values_total",
        "ocr_value_hits",
        "opf_value_hits",
        "privacy_values_total",
        "privacy_ocr_hits",
        "privacy_opf_hits",
        "utility_values_total",
        "utility_ocr_hits",
        "utility_opf_hits",
        "ocr_value_recall",
        "end_to_end_value_recall",
        "opf_given_ocr_value_recall",
        "privacy_ocr_recall",
        "privacy_end_to_end_recall",
        "privacy_opf_given_ocr_recall",
        "utility_false_positive_rate",
        "elapsed_seconds",
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
