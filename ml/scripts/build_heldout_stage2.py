from pathlib import Path
import pandas as pd


INPUT_PATH = Path("ml/data/heldout/heldout_200.csv")
OUTPUT_PATH = Path("ml/data/heldout/heldout_200_stage2.csv")


REPLACEMENT_OTHER_ROWS = [
    {"text": "хочу поговорить с оператором", "notes": "operator_contact"},
    {"text": "переведите на человека", "notes": "operator_contact"},
    {"text": "мне нужен живой сотрудник", "notes": "operator_contact"},
    {"text": "как связаться с оператором?", "notes": "operator_contact"},
    {"text": "я не понимаю бота, дайте человека", "notes": "operator_contact"},
    {"text": "оператор нужен срочно", "notes": "operator_contact"},
    {"text": "соедените с оператором", "notes": "operator_contact_typo"},
    {"text": "чел нужен, бот не помогает", "notes": "operator_contact_slang"},
    {"text": "куда отправить жалобу?", "notes": "feedback_complaint"},
    {"text": "хочу оставить жалобу на обслуживание", "notes": "feedback_complaint"},
    {"text": "нужно оставить отзыв о магазине", "notes": "feedback_complaint"},
    {"text": "как отправить отзыв?", "notes": "feedback_complaint"},
    {"text": "есть предложение по улучшению сервиса", "notes": "feedback_complaint"},
    {"text": "как отправить предложение по работе сайта?", "notes": "feedback_complaint"},
    {"text": "хочу пожаловаться на работу магазина", "notes": "feedback_complaint"},
    {"text": "куда написать комментарий о сервисе", "notes": "feedback_complaint"},
    {"text": "хочу написать претензию", "notes": "legal_claim"},
    {"text": "как подать претензию к магазину", "notes": "legal_claim"},
    {"text": "у меня притензия по обслуживанию", "notes": "legal_claim_typo"},
    {"text": "нужен живой оператор", "notes": "operator_contact_emoji"},
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = {"id", "text", "label", "expected_route", "expected_doc_id", "notes"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    other_mask = df["label"] == "other"
    other_rows = df[other_mask].copy().sort_values("id")

    if len(other_rows) != 20:
        raise ValueError(f"Expected 20 rows for label='other', got {len(other_rows)}")

    if len(REPLACEMENT_OTHER_ROWS) != 20:
        raise ValueError("REPLACEMENT_OTHER_ROWS must contain exactly 20 items")

    replacement_df = other_rows.copy().reset_index(drop=True)

    for i, repl in enumerate(REPLACEMENT_OTHER_ROWS):
        replacement_df.loc[i, "text"] = repl["text"]
        replacement_df.loc[i, "label"] = "other"
        replacement_df.loc[i, "expected_route"] = "fallback"
        replacement_df.loc[i, "expected_doc_id"] = ""
        replacement_df.loc[i, "notes"] = repl["notes"]

    df_out = df.copy()
    df_out.loc[other_mask, :] = replacement_df.values

    # Перестраховка: сортировка по id
    df_out = df_out.sort_values("id").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"[OK] Saved Stage 2 heldout to: {OUTPUT_PATH}")
    print(f"[INFO] Shape: {df_out.shape}")

    print("\n[INFO] Label distribution:")
    print(df_out["label"].value_counts().sort_index())

    print("\n[INFO] Stage 2 'other' examples:")
    print(
        df_out[df_out["label"] == "other"][
            ["id", "text", "expected_route", "notes"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()