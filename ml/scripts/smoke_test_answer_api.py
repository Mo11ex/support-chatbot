import json
import time
import requests
import pandas as pd


API_URL = "http://127.0.0.1:8001/api/v1/answer"

TEST_CASES = [
    # FAQ-like queries
    {"text": "Можно ли оформить самовывоз?", "expected_branch": "faq"},
    {"text": "Есть ли оплата при получении?", "expected_branch": "faq"},
    {"text": "Почему промокод не работает?", "expected_branch": "faq"},
    {"text": "Какая гарантия на товар?", "expected_branch": "faq"},
    {"text": "Как оформить возврат товара?", "expected_branch": "faq"},

    # RAG-like queries
    {"text": "не могу войти в аккаунт", "expected_branch": "rag"},
    {"text": "сайт не работает, белый экран", "expected_branch": "rag"},
    {"text": "какие размеры доступны?", "expected_branch": "rag"},
    {"text": "деньги списались но заказ не создался", "expected_branch": "rag"},
    {"text": "можно ли отменить заказ?", "expected_branch": "rag"},

    # Order status
    {"text": "где мой заказ?", "expected_branch": "need_order_id"},
    {"text": "статус заказа 123456", "expected_branch": "orders_api"},
    {"text": "где мой заказ 78901234", "expected_branch": "orders_api"},

    # Fallback / operator
    {"text": "нужен оператор", "expected_branch": "fallback"},
    {"text": "хочу поговорить с человеком", "expected_branch": "fallback"},
    {"text": "ыыы непонятно что", "expected_branch": "fallback"},

    # Short queries
    {"text": "скидка", "expected_branch": "faq_or_rag"},
    {"text": "возврат", "expected_branch": "faq_or_rag"},
    {"text": "ошибка", "expected_branch": "rag_or_fallback"},
    {"text": "доставка", "expected_branch": "faq_or_rag"},
]


def run_test(test_case: dict) -> dict:
    payload = {"text": test_case["text"]}

    try:
        t0 = time.perf_counter()
        resp = requests.post(API_URL, json=payload, timeout=30)
        latency = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            return {
                "text": test_case["text"],
                "expected": test_case["expected_branch"],
                "status": resp.status_code,
                "error": resp.text[:200],
            }

        data = resp.json()
        return {
            "text": test_case["text"],
            "expected": test_case["expected_branch"],
            "branch": data.get("branch"),
            "intent": data.get("intent"),
            "confidence": data.get("intent_confidence"),
            "source_type": data.get("source_type"),
            "source_id": str(data.get("source_id", ""))[:40],
            "faq_score": data.get("faq_score"),
            "rag_score": data.get("rag_score"),
            "fallback": data.get("fallback_recommended"),
            "answer": str(data.get("answer", ""))[:80],
            "api_latency_ms": data.get("latency_ms"),
            "e2e_latency_ms": round(latency, 1),
        }

    except Exception as e:
        return {
            "text": test_case["text"],
            "expected": test_case["expected_branch"],
            "error": str(e)[:200],
        }


def main():
    print(f"[INFO] Testing {len(TEST_CASES)} cases against {API_URL}\n")

    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}/{len(TEST_CASES)}] {tc['text'][:50]}...", end=" ")
        result = run_test(tc)
        results.append(result)

        if "error" in result:
            print(f"ERROR: {result.get('error', '')[:60]}")
        else:
            print(f"→ branch={result['branch']}, intent={result['intent']}, faq={result.get('faq_score')}, rag={result.get('rag_score')}")

    df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 100)
    print("[SUMMARY]")
    print("=" * 100)

    if "branch" in df.columns:
        print("\nBranch distribution:")
        print(df["branch"].value_counts())

    if "intent" in df.columns:
        print("\nIntent distribution:")
        print(df["intent"].value_counts())

    if "source_type" in df.columns:
        print("\nSource type distribution:")
        print(df["source_type"].value_counts())

    if "fallback" in df.columns:
        print(f"\nFallback recommended: {df['fallback'].sum()} / {len(df)}")

    if "e2e_latency_ms" in df.columns:
        lat = df["e2e_latency_ms"].dropna()
        if len(lat) > 0:
            print(f"\nLatency (e2e):")
            print(f"  P50: {lat.quantile(0.5):.0f} ms")
            print(f"  P95: {lat.quantile(0.95):.0f} ms")
            print(f"  P99: {lat.quantile(0.99):.0f} ms")
            print(f"  Max: {lat.max():.0f} ms")

    # Detailed table
    print("\n" + "=" * 100)
    print("[DETAILED RESULTS]")
    print("=" * 100)

    display_cols = [
        c for c in [
            "text", "expected", "branch", "intent", "confidence",
            "source_type", "faq_score", "rag_score", "fallback",
            "e2e_latency_ms", "answer",
        ]
        if c in df.columns
    ]
    print(df[display_cols].to_string(index=False))

    # Save
    output_path = "ml/logs/reports/stage6_smoke_test_results.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[OK] Saved results to: {output_path}")


if __name__ == "__main__":
    main()