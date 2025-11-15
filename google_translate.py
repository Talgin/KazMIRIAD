import time
import random
import os
import pandas as pd
from deep_translator import GoogleTranslator

# checking the PR check automation on n8n

ERROR_FILE = "./data/errors.csv"

def safe_translate_batch(translator, texts):
    """Try batch translation, fall back to single, return None on failure."""
    try:
        return translator.translate_batch(texts)
    except:
        results = []
        for t in texts:
            try:
                results.append(translator.translate(t))
                time.sleep(0.5)
            except:
                results.append(None)
        return results

def get_last_id(file):
    """Return last qa_id from CSV if exists, else None."""
    if os.path.exists(file) and os.path.getsize(file) > 0:
        df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip", engine="python")
        if not df.empty and "qa_id" in df.columns:
            return df["qa_id"].iloc[-1]
    return None

def translate_csv_with_resume(input_file, output_ru, output_kk, batch_size=50, save_interval=1000):
    df = pd.read_csv(input_file, encoding="utf-8", on_bad_lines="skip", engine="python")

    # Resume logic based on qa_id
    kk_last = get_last_id(output_kk)
    ru_last = get_last_id(output_ru)

    if kk_last is not None and ru_last is not None:
        if kk_last == ru_last:
            last_done_id = kk_last
        else:
            last_done_id = max(kk_last, ru_last)
    elif kk_last is not None:
        last_done_id = kk_last
    elif ru_last is not None:
        last_done_id = ru_last
    else:
        last_done_id = None

    if last_done_id is not None:
        df = df[df["qa_id"] > last_done_id]
        print(f"Resuming after QA_ID: {last_done_id}")
    else:
        print("Starting fresh translation")

    # Load existing or create new dataframes
    df_ru = pd.read_csv(output_ru, encoding="utf-8") if os.path.exists(output_ru) else pd.DataFrame(columns=df.columns)
    df_kk = pd.read_csv(output_kk, encoding="utf-8") if os.path.exists(output_kk) else pd.DataFrame(columns=df.columns)
    errors_df = pd.read_csv(ERROR_FILE, encoding="utf-8") if os.path.exists(ERROR_FILE) else pd.DataFrame(columns=df.columns)

    translator_ru = GoogleTranslator(source='en', target='ru')
    translator_kk = GoogleTranslator(source='en', target='kk')

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end].copy()

        batch_q = batch_df["question"].astype(str).tolist()
        batch_a = batch_df["answer"].astype(str).tolist()

        q_ru = safe_translate_batch(translator_ru, batch_q)
        a_ru = safe_translate_batch(translator_ru, batch_a)
        q_kk = safe_translate_batch(translator_kk, batch_q)
        a_kk = safe_translate_batch(translator_kk, batch_a)

        for i in range(len(batch_df)):
            row = batch_df.iloc[[i]].copy()
            failed = False

            if q_ru[i] and a_ru[i]:
                row_ru = row.copy()
                row_ru["question"] = q_ru[i]
                row_ru["answer"] = a_ru[i]
                df_ru = pd.concat([df_ru, row_ru], ignore_index=True)
            else:
                failed = True

            if q_kk[i] and a_kk[i]:
                row_kk = row.copy()
                row_kk["question"] = q_kk[i]
                row_kk["answer"] = a_kk[i]
                df_kk = pd.concat([df_kk, row_kk], ignore_index=True)
            else:
                failed = True

            if failed:
                errors_df = pd.concat([errors_df, row], ignore_index=True)

        if ((len(df_ru) % save_interval) == 0) or (end >= len(df)):
            df_ru.to_csv(output_ru, index=False, encoding="utf-8")
            df_kk.to_csv(output_kk, index=False, encoding="utf-8")
            errors_df.to_csv(ERROR_FILE, index=False, encoding="utf-8")
            print(f"Saved progress: RU={len(df_ru)}, KK={len(df_kk)}, ERR={len(errors_df)}")

        time.sleep(random.randint(5, 15))

    df_ru.to_csv(output_ru, index=False, encoding="utf-8")
    df_kk.to_csv(output_kk, index=False, encoding="utf-8")
    errors_df.to_csv(ERROR_FILE, index=False, encoding="utf-8")
    print(f"Translation complete. RU={len(df_ru)}, KK={len(df_kk)}, ERR={len(errors_df)}")

if __name__ == "__main__":
    translate_csv_with_resume(
        input_file="./data/cpart2.csv",
        output_ru="./data/combined_ru.csv",
        output_kk="./data/combined_kk.csv",
        batch_size=50,
        save_interval=1000
    )
