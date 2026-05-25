#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models/openai-privacy-filter/onnx/model_q4_embedded.onnx"
DEFAULT_TOKENIZER = REPO_ROOT / "models/openai-privacy-filter/tokenizer.json"
DEFAULT_CONFIG = REPO_ROOT / "models/openai-privacy-filter/config.json"
DEFAULT_BIN = REPO_ROOT / "target/release/safeclipper"
UNKNOWN_ANSWERS = {"", "unknown", "n/a", "not visible", "redacted", "unreadable"}
DEFAULT_GEMINI_MODEL = "google/gemini-3-flash-preview"

QA_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "answer": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence": {"type": "string"},
                },
                "required": ["id", "answer", "confidence", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

QA_SYSTEM_PROMPT = (
    "You answer open-ended questions using only the supplied screenshot. If the answer is "
    "hidden, unreadable, redacted, absent, or not directly visible, answer exactly UNKNOWN. "
    "Do not infer from world knowledge, page templates, or prior assumptions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run safeclipper over paperboy-ai/desktop-pii-210 and score deterministic QA visibility."
    )
    parser.add_argument("--dataset-name", default="paperboy-ai/desktop-pii-210")
    parser.add_argument("--split", default="train")
    parser.add_argument("--local-dataset-dir", default=None, help="Optional local desktop-pii-210 directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--safeclipper-bin", default=str(DEFAULT_BIN))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--provider", choices=["cpu", "coreml"], default="cpu")
    parser.add_argument("--ocr-backend", choices=["auto", "vision", "tesseract"], default="auto")
    parser.add_argument("--tesseract-bin", default="tesseract")
    parser.add_argument("--mask-padding", type=int, default=2)
    parser.add_argument("--qa-mode", choices=["deterministic", "gemini"], default="gemini")
    parser.add_argument("--qa-model-slug", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults to OPENROUTER_BASE_URL, OPENAI_BASE_URL, or LLM_BASE_URL.",
    )
    parser.add_argument("--llm-concurrency", type=int, default=20)
    parser.add_argument("--llm-timeout", type=float, default=None)
    parser.add_argument("--llm-image-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--llm-jpeg-quality", type=int, default=85)
    parser.add_argument("--run-dir", default=str(REPO_ROOT / "eval/desktop-pii/runs/safeclipper-desktop-pii-210"))
    parser.add_argument("--summary-json", default=str(REPO_ROOT / "eval/desktop-pii/results/safeclipper-desktop-pii-210-summary.json"))
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-image safeclipper JSON responses.")
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.local_dataset_dir:
        dataset_dir = Path(args.local_dataset_dir)
        dataset = load_dataset(
            "parquet",
            data_files=str(dataset_dir / "data" / f"{args.split}.parquet"),
            split=args.split,
        )
    else:
        dataset = load_dataset(args.dataset_name, split=args.split)

    rows = [dict(row) for row in dataset]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def normalize(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def tokens(value: Any) -> set[str]:
    return {token for token in normalize(value).split(" ") if token}


def is_unknown(answer: Any) -> bool:
    normalized = normalize(answer)
    return normalized in UNKNOWN_ANSWERS or normalized.startswith("unknown ")


def similarity(expected: Any, answer: Any) -> float:
    expected_tokens = tokens(expected)
    answer_tokens = tokens(answer)
    if not expected_tokens or not answer_tokens:
        return 0.0
    return len(expected_tokens & answer_tokens) / len(expected_tokens | answer_tokens)


def grade_answer(expected: str, answer: str) -> str:
    if is_unknown(answer):
        return "unknown"
    normalized_expected = normalize(expected)
    normalized_answer = normalize(answer)
    if normalized_expected == normalized_answer:
        return "correct"
    if len(normalized_expected) >= 4 and (
        normalized_expected in normalized_answer or normalized_answer in normalized_expected
    ):
        return "correct"
    if similarity(expected, answer) >= 0.5:
        return "partial"
    return "incorrect"


def value_found(expected: str, haystack: str) -> bool:
    expected_c = compact(expected)
    haystack_c = compact(haystack)
    if not expected_c or not haystack_c:
        return False
    if expected_c in haystack_c or haystack_c in expected_c:
        return True
    if len(expected_c) >= 12:
        return expected_c[: max(8, int(len(expected_c) * 0.75))] in haystack_c
    return False


def matched_labels(expected: str, spans: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for span in spans:
        span_text = str(span.get("text", ""))
        if value_found(expected, span_text) or value_found(span_text, expected):
            labels.append(str(span.get("label", "")))
    return labels


def safe_name(row: dict[str, Any], ordinal: int) -> str:
    file_name = str(row.get("file_name") or f"image-{ordinal:03d}.png")
    name = Path(file_name).name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def save_input_image(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = row["image"]
    if hasattr(image, "save"):
        image.save(path)
        return
    raise TypeError(f"Unsupported image value from dataset: {type(image)!r}")


def run_safeclipper(args: argparse.Namespace, image_path: Path, output_image: Path) -> tuple[dict[str, Any], float]:
    command = [
        args.safeclipper_bin,
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--config",
        args.config,
        "--image",
        str(image_path),
        "--output-image",
        str(output_image),
        "--ocr-backend",
        args.ocr_backend,
        "--tesseract-bin",
        args.tesseract_bin,
        "--mask-padding",
        str(args.mask_padding),
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(
            f"safeclipper failed for {image_path} with exit {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(proc.stdout), elapsed_ms


def questions_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    questions = []
    for question in row.get("questions", []):
        question_id = question.get("id") or question.get("question_id")
        questions.append(
            {
                "id": str(question_id),
                "category": question.get("category"),
                "question": question.get("question"),
                "expected_answer": question.get("expected_answer"),
                "rationale": question.get("rationale"),
            }
        )
    return questions


def deterministic_predictions_for_row(row: dict[str, Any], spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for question in questions_from_row(row):
        expected_answer = str(question.get("expected_answer", ""))
        labels = matched_labels(expected_answer, spans)
        blocked = bool(labels)
        predictions.append(
            {
                "question_id": question["id"],
                "image_id": row.get("image_id"),
                "scenario_id": row.get("scenario_id"),
                "domain": row.get("domain"),
                "privacy_density": row.get("privacy_density"),
                "category": question.get("category"),
                "question": question.get("question"),
                "expected_answer": expected_answer,
                "answer": "UNKNOWN" if blocked else expected_answer,
                "blocked_by_safeclipper": blocked,
                "matched_labels": labels,
                "answer_source": "safeclipper_span_proxy",
            }
        )
    return predictions


def image_to_data_url(path: Path, *, image_format: str, jpeg_quality: int) -> str:
    from PIL import Image

    with Image.open(path) as image:
        buffer = BytesIO()
        if image_format == "jpeg":
            image.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            mime_type = "image/jpeg"
        else:
            image.convert("RGB").save(buffer, format="PNG", optimize=True)
            mime_type = "image/png"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def answer_prompt(questions: list[dict[str, Any]]) -> str:
    payload = [
        {
            "id": question["id"],
            "category": question.get("category"),
            "question": question.get("question"),
        }
        for question in questions
    ]
    return (
        "Answer these open-ended questions from the screenshot only. "
        "Return a concise answer. If the answer is obscured by redaction, hidden, too small to read, absent, "
        "or would require guessing, answer exactly UNKNOWN. Questions:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def load_llm_config(args: argparse.Namespace) -> tuple[str, str]:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY or OPENAI_API_KEY is not set. Copy .env or set it in the process environment.")
    base_url = (
        args.llm_base_url
        or os.getenv("OPENROUTER_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_BASE_URL")
    )
    if not base_url:
        raise RuntimeError("Set --llm-base-url, OPENROUTER_BASE_URL, OPENAI_BASE_URL, or LLM_BASE_URL for Gemini QA mode.")
    return api_key, base_url


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


async def answer_one_with_llm(
    client: Any,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    row: dict[str, Any],
    redacted_path: Path,
) -> tuple[str, list[dict[str, Any]], float]:
    questions = questions_from_row(row)
    if not questions:
        return str(row.get("image_id")), [], 0.0

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            async with semaphore:
                started = time.perf_counter()
                response = await client.chat.completions.create(
                    model=args.qa_model_slug,
                    messages=[
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": answer_prompt(questions)},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_to_data_url(
                                            redacted_path,
                                            image_format=args.llm_image_format,
                                            jpeg_quality=args.llm_jpeg_quality,
                                        ),
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "qa_answers",
                            "strict": True,
                            "schema": QA_ANSWER_SCHEMA,
                        },
                    },
                    extra_body={"provider": {"require_parameters": True}},
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            content = content_to_text(response.choices[0].message.content)
            parsed = json.loads(content)
            answers = parsed.get("answers", [])
            return str(row.get("image_id")), answers, elapsed_ms
        except Exception as exc:  # noqa: BLE001 - retry and surface the last provider error.
            last_error = exc
            if attempt == 3:
                raise
            await asyncio.sleep(float(attempt))
    raise RuntimeError("unreachable") from last_error


async def answer_redacted_images_with_llm(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from openai import AsyncOpenAI

    api_key, base_url = load_llm_config(args)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.llm_timeout,
    )
    semaphore = asyncio.Semaphore(args.llm_concurrency)
    jobs = [
        answer_one_with_llm(client, semaphore, args, row, run_dir / str(record["redacted_image"]))
        for row, record in zip(rows, image_records, strict=True)
    ]
    results = await asyncio.gather(*jobs)
    await client.close()

    answers_by_image = {image_id: (answers, latency_ms) for image_id, answers, latency_ms in results}
    predictions: list[dict[str, Any]] = []
    latency_samples: list[float] = []
    for row in rows:
        image_id = str(row.get("image_id"))
        answers, latency_ms = answers_by_image.get(image_id, ([], 0.0))
        latency_samples.append(latency_ms)
        answer_by_id = {str(answer.get("id") or answer.get("question_id")): answer for answer in answers}
        for question in questions_from_row(row):
            answer = answer_by_id.get(question["id"], {})
            predictions.append(
                {
                    "question_id": question["id"],
                    "image_id": row.get("image_id"),
                    "scenario_id": row.get("scenario_id"),
                    "domain": row.get("domain"),
                    "privacy_density": row.get("privacy_density"),
                    "category": question.get("category"),
                    "question": question.get("question"),
                    "expected_answer": str(question.get("expected_answer", "")),
                    "answer": str(answer.get("answer", "UNKNOWN")),
                    "blocked_by_safeclipper": is_unknown(answer.get("answer", "UNKNOWN")),
                    "matched_labels": [],
                    "answer_source": "gemini_redacted_image_qa",
                    "confidence": answer.get("confidence"),
                    "evidence": answer.get("evidence"),
                }
            )
    return predictions, {
        "qa_model_slug": args.qa_model_slug,
        "llm_base_url": base_url,
        "llm_concurrency": args.llm_concurrency,
        "count": len(latency_samples),
        "mean_answer_ms": sum(latency_samples) / len(latency_samples) if latency_samples else 0.0,
    }


def score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_density: dict[str, Counter[str]] = defaultdict(Counter)

    for row in predictions:
        category = str(row["category"])
        verdict = grade_answer(str(row["expected_answer"]), str(row["answer"]))
        leaks_sensitive_value = category == "privacy" and verdict in {"correct", "partial"}
        row["verdict"] = verdict
        row["leaks_sensitive_value"] = leaks_sensitive_value

        counters[f"{category}_total"] += 1
        counters[f"{category}_{verdict}"] += 1
        if leaks_sensitive_value:
            counters["privacy_leaks"] += 1
        if row["blocked_by_safeclipper"]:
            counters[f"{category}_blocked"] += 1

        for bucket in (by_domain[str(row.get("domain") or "unknown")], by_density[str(row.get("privacy_density") or "unknown")]):
            bucket[f"{category}_total"] += 1
            if leaks_sensitive_value:
                bucket["privacy_leaks"] += 1
            if category == "utility" and verdict == "correct":
                bucket["utility_correct"] += 1
            if row["blocked_by_safeclipper"]:
                bucket[f"{category}_blocked"] += 1

    def summarize(counter: Counter[str]) -> dict[str, Any]:
        utility_total = counter["utility_total"]
        privacy_total = counter["privacy_total"]
        utility_correct = counter["utility_correct"]
        utility_partial = counter["utility_partial"]
        privacy_leaks = counter["privacy_leaks"]
        utility_score = utility_correct / utility_total if utility_total else 0.0
        utility_soft_score = (utility_correct + 0.5 * utility_partial) / utility_total if utility_total else 0.0
        privacy_leak_rate = privacy_leaks / privacy_total if privacy_total else 0.0
        privacy_block_rate = 1.0 - privacy_leak_rate if privacy_total else 0.0
        return {
            "question_total": utility_total + privacy_total,
            "utility_total": utility_total,
            "utility_correct": utility_correct,
            "utility_partial": utility_partial,
            "utility_blocked_by_safeclipper": counter["utility_blocked"],
            "utility_score": utility_score,
            "utility_soft_score": utility_soft_score,
            "privacy_total": privacy_total,
            "privacy_leaks": privacy_leaks,
            "privacy_blocked_by_safeclipper": counter["privacy_blocked"],
            "privacy_leak_rate": privacy_leak_rate,
            "privacy_block_rate": privacy_block_rate,
            "overall_score": 0.5 * utility_score + 0.5 * privacy_block_rate,
        }

    return {
        "summary": summarize(counters),
        "by_domain": {key: summarize(value) for key, value in sorted(by_domain.items())},
        "by_privacy_density": {key: summarize(value) for key, value in sorted(by_density.items())},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    run_dir = Path(args.run_dir)
    originals_dir = run_dir / "originals"
    redacted_dir = run_dir / "redacted"
    responses_dir = run_dir / "safeclipper-json"
    for path in [originals_dir, redacted_dir, responses_dir, Path(args.summary_json).parent]:
        path.mkdir(parents=True, exist_ok=True)

    for required in [args.safeclipper_bin, args.model, args.tokenizer, args.config]:
        if not Path(required).exists():
            raise FileNotFoundError(required)

    rows = load_rows(args)
    deterministic_predictions: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    latency_samples: list[float] = []
    model_latency_samples: list[float] = []
    mask_counts: list[int] = []
    ocr_token_counts: list[int] = []

    for ordinal, row in enumerate(rows):
        image_name = safe_name(row, ordinal)
        image_path = originals_dir / image_name
        redacted_path = redacted_dir / image_name
        response_path = responses_dir / f"{Path(image_name).stem}.json"
        save_input_image(row, image_path)

        if args.resume and response_path.exists() and redacted_path.exists():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            elapsed_ms = float(response.get("_wall_latency_ms", 0.0))
        else:
            response, elapsed_ms = run_safeclipper(args, image_path, redacted_path)
            response["_wall_latency_ms"] = elapsed_ms
            response_path.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")

        spans = list(response.get("detected_spans", []))
        image_redaction = response.get("image_redaction") or {}
        latency_samples.append(elapsed_ms)
        model_latency_samples.append(float(response.get("summary", {}).get("latency_ms", 0.0)))
        mask_counts.append(int(image_redaction.get("mask_count") or 0))
        ocr_token_counts.append(int(image_redaction.get("ocr_token_count") or 0))

        image_record = {
            "ordinal": ordinal,
            "image_id": row.get("image_id"),
            "scenario_id": row.get("scenario_id"),
            "domain": row.get("domain"),
            "privacy_density": row.get("privacy_density"),
            "original_image": str(image_path.relative_to(run_dir)),
            "redacted_image": str(redacted_path.relative_to(run_dir)),
            "response_json": str(response_path.relative_to(run_dir)),
            "span_count": len(spans),
            "mask_count": image_redaction.get("mask_count"),
            "ocr_token_count": image_redaction.get("ocr_token_count"),
            "wall_latency_ms": elapsed_ms,
            "model_latency_ms": response.get("summary", {}).get("latency_ms"),
        }
        image_records.append(image_record)

        deterministic_predictions.extend(deterministic_predictions_for_row(row, spans))

        if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(rows):
            print(f"processed {ordinal + 1}/{len(rows)}", file=sys.stderr, flush=True)

    write_jsonl(run_dir / "images.jsonl", image_records)
    write_jsonl(run_dir / "span_proxy_predictions.jsonl", deterministic_predictions)

    llm_summary: dict[str, Any] | None = None
    if args.qa_mode == "gemini":
        predictions, llm_summary = asyncio.run(answer_redacted_images_with_llm(args, rows, image_records, run_dir))
    else:
        predictions = deterministic_predictions

    write_jsonl(run_dir / "predictions.jsonl", predictions)

    scored = score_predictions(predictions)
    summary = {
        "run_name": run_dir.name,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": args.dataset_name,
            "split": args.split,
            "row_count": len(rows),
            "question_count": len(predictions),
        },
        "settings": {
            "safeclipper_bin": str(Path(args.safeclipper_bin).resolve()),
            "model": str(Path(args.model).resolve()),
            "provider": args.provider,
            "ocr_backend": args.ocr_backend,
            "mask_padding": args.mask_padding,
            "qa_mode": args.qa_mode,
        },
        "latency": {
            "count": len(latency_samples),
            "mean_wall_ms": sum(latency_samples) / len(latency_samples) if latency_samples else 0.0,
            "mean_model_ms": sum(model_latency_samples) / len(model_latency_samples) if model_latency_samples else 0.0,
        },
        "llm": llm_summary,
        "image_redaction": {
            "total_masks": sum(mask_counts),
            "mean_masks_per_image": sum(mask_counts) / len(mask_counts) if mask_counts else 0.0,
            "mean_ocr_tokens_per_image": sum(ocr_token_counts) / len(ocr_token_counts) if ocr_token_counts else 0.0,
        },
        "results": scored["summary"],
        "by_domain": scored["by_domain"],
        "by_privacy_density": scored["by_privacy_density"],
        "artifacts": {
            "run_dir": str(run_dir),
            "predictions_jsonl": str(run_dir / "predictions.jsonl"),
            "span_proxy_predictions_jsonl": str(run_dir / "span_proxy_predictions.jsonl"),
            "images_jsonl": str(run_dir / "images.jsonl"),
            "redacted_images": str(redacted_dir),
            "safeclipper_json": str(responses_dir),
        },
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary["results"], indent=2))
    print(f"wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
