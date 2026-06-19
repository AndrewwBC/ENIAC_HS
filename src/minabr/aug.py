import json
import random
import logging
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

from src.minabr.system_prompt import SYSTEM_PROMPT
from src.minabr.use_original_data import train

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_NAME  = "huihui-ai/Qwen2.5-14B-Instruct-abliterated"
MAX_WORKERS = 200
OUTPUT_FILE = "./augmented/minabr_augmented.csv"

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Carrega o DataFrame original apenas para calcular a meta de balanceamento
original_df = pd.read_csv("./dataset/minabr.csv")
text_col = "comment" if "comment" in train.columns else "text"

hate_train_pool = train[
    (train["label"] == 1)
][text_col].tolist()

def build_few_shot_block() -> tuple[str, list[str]]:
    # Sorteia do pool gerado a partir do dataset de treino
    samples = random.sample(hate_train_pool, min(5, len(hate_train_pool)))
    block   = "\n".join(f"- {s}" for s in samples)
    return block, samples

def generate(_: int) -> tuple[str, list[str]] | None:
    few_shot_block, few_shot_samples = build_few_shot_block()
    system = f"{SYSTEM_PROMPT}\n\nExemplos de comentários reais:\n{few_shot_block}"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": "Gere um novo comentário de ódio baseado nos exemplos."},
            ],
            temperature=0.95,
            top_p=0.8,
            presence_penalty=1,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )
        text = resp.choices[0].message.content.strip()
        if not text:
            log.warning("Modelo retornou resposta vazia.")
            return None
        if resp.choices[0].finish_reason == "length":
            log.warning("Resposta truncada por limite de tokens: %s", text[:60])
            return None
        return text, few_shot_samples
    except Exception as e:
        log.error("Erro ao gerar comentário: %s", e)
        return None


# ── balanceamento (Cálculo baseado no original_df) ───────────────────────────
counts = original_df["odio"].value_counts()
n_hate = 1751 - counts.get(1, 0)  # Usa .get() por segurança caso venha zerado

log.info("Distribuição original: %s", counts.to_dict())
log.info("A gerar: %d exemplos de hate.", n_hate)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(executor.map(generate, range(n_hate)))

ok = [r for r in results if r is not None]
log.info("%d/%d gerados com sucesso.", len(ok), n_hate)

generated_rows: list[dict] = []
for text, shots in ok:
    generated_rows.append({"text": text, "odio": 1, "shots": shots})

with open("./augmented/generated.json", "w", encoding="utf-8") as f:
    json.dump(generated_rows, f, ensure_ascii=False, indent=2)
log.info("Gerados salvos em generated.json (%d exemplos)", len(generated_rows))

# ── merge e save ───────────────────────────────────────────────────────────────
# Garante o alinhamento de colunas com o original_df para o concat
generated_df = pd.DataFrame([{text_col: r["text"], "odio": r["odio"]} for r in generated_rows])
final_df     = pd.concat([original_df, generated_df], ignore_index=True).sample(frac=1, random_state=42)

final_df.to_csv(OUTPUT_FILE, index=False)
log.info("Salvo em %s  (%d linhas totais)", OUTPUT_FILE, len(final_df))