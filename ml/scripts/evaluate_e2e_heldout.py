import json
import time
import requests
import pandas as pd
from pathlib import Path

API_URL = "http://127.0.0.1:8001/api/v1/answer"
HELDOUT_PATH = Path("ml/data/heldout/heldout_200_stage2.csv")
OUTPUT_DIR = Path("ml/logs/reports/stage8_e2e_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Маппинг label -> ожидаемая ветка оркестратора
LABEL_TO_EXPECTED_BRANCH = {
    "order_status": "need_order_id",       # без номера заказа бот попросит номер
    "delivery": "faq_direct",
    "payment_refund": "faq_direct",
    "return_exchange": "faq_direct",
    "promo_loyalty": "faq_direct",
    "account": "rag_with_filter",
    "product_info": "rag_with_filter",
    "technical_issue": "rag_with_filter",
    "general_info": "fallback",
    "other": "fallback",
}

# Какие ветки считаем "правильным ответом" для каждого label
LABEL_TO_ACCEPTABLE_BRANCHES = {
    "order_status": {"need_order_id", "orders_api", "orders_not_found", "rag_with_filter"},
    "delivery": {"faq_direct", "rag_with_filter"},
    "payment_refund": {"faq_direct", "rag_with_filter"},
    "return_exchange": {"faq_direct", "rag_with_filter"},
    "promo_loyalty": {"faq_direct", "rag_with_filter"},
    "account": {"rag_with_filter", "rag_no_filter", "faq_direct"},
    "product_info": {"rag_with_filter", "rag_no_filter", "faq_direct"},
    "technical_issue": {"rag_with_filter", "rag_no_filter", "faq_direct"},
    "general_info": {"fallback", "rag_with_filter", "rag_no_filter", "faq_direct"},
    "other": {"fallback"},
}


def run_query(text: str) -> dict:
    try:
        t0 = time.perf_counter()
        resp = requests.post(API_URL, json={"text": text}, timeout=30)
        latency = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            return {
                "status": "error",
                "status_code": resp.status_code,
                "error": resp.text[:200],
                "latency_ms": round(latency, 1),
            }

        data = resp.json()
        data["status"] = "ok"
        data["latency_ms"] = round(latency, 1)
        return data

    except Exception as e:
        return {
            "status": "exception",
            "error": str(e)[:200],
            "latency_ms": 0.0,
        }


def main():
    if not HELDOUT_PATH.exists():
        raise FileNotFoundError(f"Heldout file not found: {HELDOUT_PATH}")

    heldout = pd.read_csv(HELDOUT_PATH)
    print(f"[INFO] Loaded {len(heldout)} heldout queries from {HELDOUT_PATH}")

    results = []

    for i, row in heldout.iterrows():
        text = row["text"]
        true_label = row["label"]
        expected_route = row.get("expected_route", "")

        print(f"[{i+1:03d}/{len(heldout)}] {text[:50]}...", end=" ")

        response = run_query(text)

        pred_intent = response.get("intent")
        pred_branch = response.get("branch")
        pred_answer = str(response.get("answer", ""))[:200]
        latency = response.get("latency_ms", 0.0)
        status = response.get("status", "unknown")

        # Проверяем корректность
        intent_correct = (pred_intent == true_label) if pred_intent else False

        acceptable = LABEL_TO_ACCEPTABLE_BRANCHES.get(true_label, set())
        branch_acceptable = pred_branch in acceptable if pred_branch else False

        has_answer = bool(response.get("answer"))
        fallback_recommended = response.get("fallback_recommended", False)

        print(f"→ intent={pred_intent}, branch={pred_branch}, ok={intent_correct}, lat={latency:.0f}ms")

        results.append({
            "id": row.get("id", i),
            "text": text,
            "true_label": true_label,
            "expected_route": expected_route,
            "pred_intent": pred_intent,
            "pred_branch": pred_branch,
            "intent_correct": intent_correct,
            "branch_acceptable": branch_acceptable,
            "has_answer": has_answer,
            "fallback_recommended": fallback_recommended,
            "answer_preview": pred_answer,
            "latency_ms": latency,
            "status": status,
        })

    df = pd.DataFrame(results)

    # ── Метрики ──────────────────────────────────────────────────────
    total = len(df)
    ok_status = df[df["status"] == "ok"]
    total_ok = len(ok_status)

    intent_accuracy = ok_status["intent_correct"].mean() if total_ok > 0 else 0.0
    branch_accuracy = ok_status["branch_acceptable"].mean() if total_ok > 0 else 0.0
    answer_rate = ok_status["has_answer"].mean() if total_ok > 0 else 0.0
    fallback_rate = ok_status["fallback_recommended"].mean() if total_ok > 0 else 0.0

    latencies = ok_status["latency_ms"].dropna()
    lat_p50 = latencies.quantile(0.5) if len(latencies) > 0 else 0.0
    lat_p95 = latencies.quantile(0.95) if len(latencies) > 0 else 0.0
    lat_p99 = latencies.quantile(0.99) if len(latencies) > 0 else 0.0
    lat_max = latencies.max() if len(latencies) > 0 else 0.0

    # Per-class intent accuracy
    per_class_intent = (
        ok_status.groupby("true_label")["intent_correct"]
        .mean()
        .sort_index()
        .to_dict()
    )

    # Per-class branch accuracy
    per_class_branch = (
        ok_status.groupby("true_label")["branch_acceptable"]
        .mean()
        .sort_index()
        .to_dict()
    )

    # Branch distribution
    branch_dist = ok_status["pred_branch"].value_counts().to_dict()

    # Errors by true label
    errors = ok_status[~ok_status["intent_correct"]].copy()
    error_pairs = (
        errors.groupby(["true_label", "pred_intent"])
        .size()
        .sort_values(ascending=False)
        .head(20)
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    summary = {
        "total_queries": total,
        "successful_queries": total_ok,
        "failed_queries": total - total_ok,
        "intent_accuracy": round(intent_accuracy, 4),
        "branch_acceptable_rate": round(branch_accuracy, 4),
        "answer_rate": round(answer_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "latency": {
            "p50_ms": round(lat_p50, 1),
            "p95_ms": round(lat_p95, 1),
            "p99_ms": round(lat_p99, 1),
            "max_ms": round(lat_max, 1),
        },
        "per_class_intent_accuracy": per_class_intent,
        "per_class_branch_acceptable_rate": per_class_branch,
        "branch_distribution": branch_dist,
        "top_intent_errors": error_pairs,
    }

    # ── Сохранение ───────────────────────────────────────────────────
    pred_path = OUTPUT_DIR / "e2e_predictions.csv"
    summary_path = OUTPUT_DIR / "e2e_summary.json"

    df.to_csv(pred_path, index=False, encoding="utf-8")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Отчёт ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[E2E EVALUATION SUMMARY]")
    print("=" * 80)

    print(f"\nTotal queries:        {total}")
    print(f"Successful:           {total_ok}")
    print(f"Failed:               {total - total_ok}")

    print(f"\nIntent accuracy:      {intent_accuracy:.4f}")
    print(f"Branch acceptable:    {branch_accuracy:.4f}")
    print(f"Answer rate:          {answer_rate:.4f}")
    print(f"Fallback rate:        {fallback_rate:.4f}")

    print(f"\nLatency:")
    print(f"  P50:  {lat_p50:.0f} ms")
    print(f"  P95:  {lat_p95:.0f} ms")
    print(f"  P99:  {lat_p99:.0f} ms")
    print(f"  Max:  {lat_max:.0f} ms")

    print("\nPer-class intent accuracy:")
    for label, acc in sorted(per_class_intent.items()):
        print(f"  {label:20s}: {acc:.4f}")

    print("\nPer-class branch acceptable rate:")
    for label, acc in sorted(per_class_branch.items()):
        print(f"  {label:20s}: {acc:.4f}")

    print("\nBranch distribution:")
    for branch, count in sorted(branch_dist.items(), key=lambda x: -x[1]):
        print(f"  {branch:20s}: {count}")

    if error_pairs:
        print("\nTop intent errors:")
        for ep in error_pairs[:10]:
            print(f"  {ep['true_label']:20s} -> {ep['pred_intent']:20s} ({ep['count']})")

    print(f"\n[OK] Saved predictions: {pred_path}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()