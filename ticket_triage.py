"""Privacy-aware support ticket triage using the supplied offline mock LLM."""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import mock_llm


ROOT = Path(__file__).resolve().parent
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUTS = ROOT / "outputs"
MAX_INPUT_TOKENS = 400
TOKEN_BUDGET_DEMO = 6000
TOKEN_BUDGET_FINAL = 100_000
MAX_RETRIES = 3
BASE_DELAY = 0.05
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")
REDACTION_PATTERNS = (
    (EMAIL_PATTERN, "[EMAIL]"),
    (PHONE_PATTERN, "[PHONE]"),
    (ORDER_PATTERN, "[ORDER_ID]"),
)

ZERO_SHOT_PROMPT = """CLASSIFY each support ticket into exactly one category: billing, delivery, technical, or fraud.
Reply with JSON only, in this form: {"category": "...", "reason": "..."}.
The category must be one of the four listed categories."""

FEW_SHOT_PROMPT = """CLASSIFY each support ticket into exactly one category: billing, delivery, technical, or fraud.
Reply with JSON only, in this form: {"category": "...", "reason": "..."}.
The category must be one of the four listed categories.
EXAMPLES:
Ticket: I was charged twice and need a refund.
Reply: {"category": "billing", "reason": "duplicate charge"}
Ticket: Someone made an unauthorized transaction; this is fraud.
Reply: {"category": "fraud", "reason": "unauthorized transaction"}
Ticket: The app crashes when I open the cart.
Reply: {"category": "technical", "reason": "app crash"}"""


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    redacted = text
    for (pattern, placeholder), name in zip(
        REDACTION_PATTERNS, ("emails", "phones", "order_ids")
    ):
        redacted, count = pattern.subn(placeholder, redacted)
        counts[name] = count
    return redacted, counts


def validate_reply(reply: str) -> tuple[str, str, str]:
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        words = re.findall(r"[a-z]+", reply.lower())
        matches = [word for word in words if word in ALLOWED_CATEGORIES]
        if len(matches) == 1:
            return matches[0], reply, "classified_keyword"
        return "", "", "flagged_invalid_output"

    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("category"), str)
        and parsed.get("category") in ALLOWED_CATEGORIES
    ):
        reason = parsed.get("reason", "")
        return parsed["category"], str(reason), "classified_json"
    return "", "", "flagged_invalid_output"


def call_with_retries(
    messages: list[dict[str, str]],
    ticket_id: str,
    usage_total: int,
    token_budget: int | None,
    retry_logs: list[str],
) -> tuple[Any | None, str]:
    retry_number = 0
    while True:
        if token_budget is not None and usage_total >= token_budget:
            return None, "budget_exhausted"
        try:
            return mock_llm.chat(messages), "called"
        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2**retry_number) * random.uniform(0.9, 1.15)
            log_line = (
                f"ticket_id={ticket_id} retry={retry_number} wait={wait:.6f}s"
            )
            print(log_line)
            retry_logs.append(log_line)
            time.sleep(wait)
            retry_number += 1


def prepare_redactions(
    tickets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    redacted_texts: list[str] = []
    totals = {"emails": 0, "phones": 0, "order_ids": 0}
    for text in tickets["text"].astype(str):
        redacted, counts = redact_text(text)
        redacted_texts.append(redacted)
        for key, count in counts.items():
            totals[key] += count
        if any(pattern.search(redacted) for pattern, _ in REDACTION_PATTERNS):
            raise AssertionError("A personal identifier remains after redaction.")
    result = tickets.copy()
    result["redacted_text"] = redacted_texts
    totals["total"] = sum(totals.values())
    return result, totals


def run_pipeline(
    tickets: pd.DataFrame,
    system_prompt: str,
    token_budget: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    mock_llm.reset()
    retry_logs: list[str] = []
    running_tokens = 0
    results: list[dict[str, object]] = []

    for row in tickets.itertuples(index=False):
        raw_text = str(row.text)
        redacted_text = str(row.redacted_text)
        base: dict[str, object] = {
            "ticket_id": row.ticket_id,
            "customer_id": row.customer_id,
            "status": "",
            "category": "",
            "reason": "",
            "tokens_used": 0,
        }
        if not raw_text.strip():
            base["status"] = "skipped_empty"
            results.append(base)
            continue
        estimated = mock_llm.est_tokens(raw_text)
        if estimated > MAX_INPUT_TOKENS:
            base["status"] = "flagged_long"
            results.append(base)
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text},
        ]
        if messages[-1]["content"] != redacted_text:
            raise AssertionError("User message must contain only redacted text.")
        if any(
            pattern.search(messages[-1]["content"])
            for pattern, _ in REDACTION_PATTERNS
        ):
            raise AssertionError("Unredacted personal data would reach the model.")

        response, call_status = call_with_retries(
            messages, str(row.ticket_id), running_tokens, token_budget, retry_logs
        )
        if call_status == "budget_exhausted":
            base["status"] = "budget_exhausted"
            results.append(base)
            continue

        running_tokens += int(response.usage.total_tokens)
        base["tokens_used"] = int(response.usage.total_tokens)
        reply = response.choices[0].message.content or ""
        category, reason, status = validate_reply(reply)
        base.update({"category": category, "reason": reason, "status": status})
        results.append(base)

    result_frame = pd.DataFrame(results)
    result_frame.attrs["total_tokens"] = running_tokens
    return result_frame, retry_logs


def accuracy_and_confusion(result: pd.DataFrame, source: pd.DataFrame) -> tuple[int, float, pd.DataFrame]:
    evaluated = result.merge(
        source[["ticket_id", "true_category"]], on="ticket_id", how="left"
    )
    classified = evaluated.loc[
        evaluated["status"].isin(["classified_json", "classified_keyword"])
    ]
    correct = int(classified["category"].eq(classified["true_category"]).sum())
    accuracy = correct / len(classified) if len(classified) else 0.0
    confusion = pd.crosstab(
        classified["true_category"],
        classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    ).reindex(
        index=["billing", "delivery", "technical", "fraud"],
        columns=["billing", "delivery", "technical", "fraud"],
        fill_value=0,
    )
    return len(classified), accuracy, confusion


def format_confusion(confusion: pd.DataFrame) -> list[str]:
    headers = ["Actual / Predicted", *confusion.columns.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for category, row in confusion.iterrows():
        lines.append(
            "| "
            + " | ".join([category, *(str(int(value)) for value in row)])
            + " |"
        )
    return lines


def add_routing(
    results: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    segment_columns = segments[["customer_id", "segment_name", "is_premium"]].copy()
    segment_columns["is_premium"] = (
        segment_columns["is_premium"].astype(str).str.lower().eq("true")
    )
    routed = results.merge(segment_columns, on="customer_id", how="left")
    routed["segment_name"] = routed["segment_name"].fillna("")
    routed["is_premium"] = routed["is_premium"].fillna(False).astype(bool)
    human_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    def route(row: pd.Series) -> tuple[str, str]:
        if row["status"] in human_statuses:
            return "human_review", "NORMAL"
        if row["category"] == "fraud":
            return "security_team", "HIGH"
        if row["is_premium"]:
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"

    routing = routed.apply(route, axis=1, result_type="expand")
    routed["route"] = routing[0]
    routed["priority"] = routing[1]
    return routed


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    tickets = pd.read_csv(TICKETS_FILE, keep_default_na=False)
    if len(tickets) != 164:
        raise AssertionError(f"Expected 164 tickets, found {len(tickets)}.")
    tickets, redaction_totals = prepare_redactions(tickets)
    if redaction_totals != {
        "emails": 55,
        "phones": 91,
        "order_ids": 132,
        "total": 278,
    }:
        raise AssertionError(f"Unexpected redaction counts: {redaction_totals}")

    random.seed(42)
    zero_shot, zero_retries = run_pipeline(tickets, ZERO_SHOT_PROMPT, None)
    random.seed(42)
    few_shot, few_retries = run_pipeline(tickets, FEW_SHOT_PROMPT, None)
    random.seed(42)
    budget_result, budget_retries = run_pipeline(
        tickets, ZERO_SHOT_PROMPT, TOKEN_BUDGET_DEMO
    )
    random.seed(42)
    final_result, final_retries = run_pipeline(
        tickets, FEW_SHOT_PROMPT, TOKEN_BUDGET_FINAL
    )

    budget_exhausted = int(budget_result["status"].eq("budget_exhausted").sum())
    if budget_exhausted == 0:
        raise AssertionError("The demo token budget did not stop any ticket.")
    final_segments = pd.read_csv(SEGMENTS_FILE, keep_default_na=False)
    final_routed = add_routing(final_result, final_segments)
    if len(final_routed) != 164:
        raise AssertionError("Final triage output must include all 164 tickets.")
    final_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]
    final_routed[final_columns].to_csv(
        OUTPUTS / "triage_results.csv", index=False
    )
    budget_result.to_csv(OUTPUTS / "triage_budget_demo.csv", index=False)
    retry_logs = zero_retries + few_retries + budget_retries + final_retries
    (OUTPUTS / "triage_retry_log.txt").write_text(
        "\n".join(retry_logs) + "\n", encoding="utf-8"
    )

    zero_count, zero_accuracy, zero_confusion = accuracy_and_confusion(
        zero_shot, tickets
    )
    few_count, few_accuracy, few_confusion = accuracy_and_confusion(
        few_shot, tickets
    )
    expected_status_counts = {
        "classified_json": 135,
        "classified_keyword": 12,
        "flagged_invalid_output": 9,
        "skipped_empty": 4,
        "flagged_long": 4,
    }
    for prompt_name, run, classified_count, accuracy in (
        ("zero-shot", zero_shot, zero_count, zero_accuracy),
        ("few-shot", few_shot, few_count, few_accuracy),
    ):
        if Counter(run["status"]) != Counter(expected_status_counts):
            raise AssertionError(
                f"Unexpected {prompt_name} status counts: {dict(Counter(run['status']))}"
            )
        if classified_count != 147:
            raise AssertionError(
                f"{prompt_name} classified {classified_count} tickets, expected 147."
            )
    if round(zero_accuracy, 3) != 0.857 or few_accuracy != 1.0:
        raise AssertionError(
            f"Unexpected prompt accuracies: zero-shot={zero_accuracy:.3f}, "
            f"few-shot={few_accuracy:.3f}"
        )
    if len(zero_retries) != 19 or len(few_retries) != 19:
        raise AssertionError(
            "A full mock run must recover all 19 transient failures by retry."
        )
    zero_tokens = int(zero_shot.attrs["total_tokens"])
    few_tokens = int(few_shot.attrs["total_tokens"])
    zero_counts = Counter(zero_shot["status"])
    few_counts = Counter(few_shot["status"])
    retry_counts = {
        "zero-shot": len(zero_retries),
        "few-shot": len(few_retries),
        "budget-demo": len(budget_retries),
        "final-few-shot": len(final_retries),
    }
    empty_count = int(tickets["text"].astype(str).str.strip().eq("").sum())
    long_count = int(
        tickets["text"].map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS).sum()
    )
    remaining_identifiers = sum(
        pattern.search(text) is not None
        for text in tickets["redacted_text"]
        for pattern, _ in REDACTION_PATTERNS
    )
    if remaining_identifiers:
        raise AssertionError("Personal identifiers remain in redacted text.")

    report = [
        "# Part C — Support ticket triage",
        "",
        "Run `python ticket_triage.py` after Part B. Tickets are processed in file order. The supplied `mock_llm.py` is offline and all prompts use redacted text only.",
        "",
        "## Input guards and redaction",
        "",
        f"- Empty/whitespace tickets skipped without a model call: **{empty_count}**.",
        f"- Tickets above {MAX_INPUT_TOKENS} estimated input tokens flagged without a model call: **{long_count}**.",
        f"- Email substitutions: **{redaction_totals['emails']}**; phone substitutions: **{redaction_totals['phones']}**; order-ID substitutions: **{redaction_totals['order_ids']}**; total: **{redaction_totals['total']}**.",
        f"- Reapplying all three redaction patterns found **{remaining_identifiers}** matches.",
        "",
        "The zero-shot and few-shot system prompts both include `CLASSIFY`, the four allowed categories, and the required JSON-only schema. Only the redacted ticket text is sent as the user-message content.",
        "",
        "## Retry behavior",
        "",
        f"Each run resets the mock immediately before processing. Retry logs are saved in `outputs/triage_retry_log.txt`; total retries across the four independent runs: **{len(retry_logs)}** ({', '.join(f'{name}: {count}' for name, count in retry_counts.items())}). There were no permanent model-call errors.",
        "",
        "## Zero-shot and few-shot comparison",
        "",
        "| Prompt | Classified | Accuracy | Correct / classified | Total tokens | Tokens per classified ticket | Status counts |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| Zero-shot | {zero_count} | {zero_accuracy:.1%} | {round(zero_accuracy * zero_count)} / {zero_count} | {zero_tokens:,} | {zero_tokens / zero_count:.2f} | {dict(zero_counts)} |",
        f"| Few-shot | {few_count} | {few_accuracy:.1%} | {round(few_accuracy * few_count)} / {few_count} | {few_tokens:,} | {few_tokens / few_count:.2f} | {dict(few_counts)} |",
        "",
        "### Zero-shot confusion table",
        "",
        *format_confusion(zero_confusion),
        "",
        f"Few-shot used {few_tokens - zero_tokens:,} additional tokens total ({few_tokens / few_count - zero_tokens / zero_count:.2f} additional tokens per classified ticket) and improved accuracy from {zero_accuracy:.1%} to {few_accuracy:.1%}. The extra prompt cost is worthwhile when avoiding a misrouted fraud or technical ticket has higher operational cost than the incremental token spend. When review capacity or token budgets are tight and lower accuracy is acceptable, the zero-shot prompt is cheaper; the measured value of few-shot depends on the cost of the errors it prevents.",
        "",
        "## Token-budget demonstration",
        "",
        f"The zero-shot demonstration used a budget of **{TOKEN_BUDGET_DEMO:,} tokens** and marked **{budget_exhausted}** tickets `budget_exhausted` without calling the model.",
        "",
        "## Final few-shot routing",
        "",
        f"The final few-shot run used a {TOKEN_BUDGET_FINAL:,}-token budget and wrote all **{len(final_routed)}** ticket rows to `outputs/triage_results.csv`. JSON/keyword classifications are rule-validated; empty, long, invalid-output and budget-exhausted tickets are routed to human review. Fraud routes to the security team; remaining valid classifications go to the Premium desk or support queue based on the Part B segment flag.",
        "",
        f"Final status counts: `{dict(Counter(final_routed['status']))}`. Human-review routes: **{int(final_routed['route'].eq('human_review').sum())}**; rule-validated classifications: **{int(final_routed['status'].isin(['classified_json', 'classified_keyword']).sum())}**.",
        "",
    ]
    (OUTPUTS / "part_c_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Redactions: {redaction_totals}")
    print(f"Guard counts: empty={empty_count}, long={long_count}")
    print(f"Zero-shot: classified={zero_count}, accuracy={zero_accuracy:.1%}, tokens={zero_tokens}")
    print(f"Few-shot: classified={few_count}, accuracy={few_accuracy:.1%}, tokens={few_tokens}")
    print(f"Budget exhausted at {TOKEN_BUDGET_DEMO}: {budget_exhausted}")
    print(f"Final output rows: {len(final_routed)}")


if __name__ == "__main__":
    main()
