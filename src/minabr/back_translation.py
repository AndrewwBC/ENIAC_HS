import json
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OUTPUT_FILE = "minabr_back_translation.csv"

original_df = pd.read_csv("./dataset/minabr.csv").iloc[:4]

text_col = "comment" if "comment" in original_df.columns else "text"
label_col = "odio" if "odio" in original_df.columns else "label"

hate_df = original_df[original_df[label_col] == 1].copy()
log.info(f"Encontradas {len(hate_df)} instâncias de ódio para retrotradução.")

log.info("Carregando modelos T5 da Unicamp-DL...")

tokenizer_pt_en = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")
model_pt_en = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5")
pten_pipeline = pipeline("text2text-generation", model=model_pt_en, tokenizer=tokenizer_pt_en, device=0)

tokenizer_en_pt = AutoTokenizer.from_pretrained("unicamp-dl/translation-en-pt-t5")
model_en_pt = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-en-pt-t5")
enpt_pipeline = pipeline("text2text-generation", model=model_en_pt, tokenizer=tokenizer_en_pt, device=0)


def back_translate(comment_text: str) -> dict | None:
    if not isinstance(comment_text, str) or not comment_text.strip():
        return None

    try:
        res_en = pten_pipeline(
            f"translate Portuguese to English: {comment_text}",
            max_length=256,
            do_sample=False,
        )
        english_text = res_en[0]["generated_text"].strip()
        if not english_text:
            return None

        res_pt = enpt_pipeline(
            f"translate English to Portuguese: {english_text}",
            max_length=256,
            do_sample=False,
        )
        back_translated_text = res_pt[0]["generated_text"].strip()
        if not back_translated_text:
            return None

        if back_translated_text.lower() == comment_text.lower():
            return None

        return {
            "original": comment_text,
            "en": english_text,
            "back_translated": back_translated_text,
        }

    except Exception as e:
        log.error(f"Erro ao traduzir '{comment_text[:30]}...': {e}")
        return None


log.info("Iniciando retrotradução...")

records = []
generated_rows = []

for idx, row in hate_df.iterrows():
    original_text = row[text_col]
    result = back_translate(original_text)

    if result:
        records.append(result)
        generated_rows.append({text_col: result["back_translated"], label_col: 1})

log.info(f"{len(generated_rows)} novas instâncias geradas.")

with open("generated_back_translation.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

generated_df = pd.DataFrame(generated_rows)
final_df = pd.concat([original_df, generated_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df.to_csv(OUTPUT_FILE, index=False)
log.info(f"Dataset salvo em '{OUTPUT_FILE}' ({len(final_df)} linhas totais).")