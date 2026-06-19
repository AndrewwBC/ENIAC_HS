import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer
from src.bert.file import BertForTextEmbedding
from base import BASE_DIR

model = BertForTextEmbedding()
model.load_state_dict(torch.load(BASE_DIR.joinpath("checkpoints/final_model/model.pt")))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def merge_subwords(tokens, scores):
    merged_tokens = []
    merged_scores = []
    
    for token, score in zip(tokens, scores):
        if token.startswith("##"):
            merged_tokens[-1] += token[2:]
            merged_scores[-1] = max(merged_scores[-1], score)
        else:
            merged_tokens.append(token)
            merged_scores.append(score)
    
    return list(zip(merged_tokens, merged_scores))

def explain(text: str, target_class: int = 1):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

    lig = LayerIntegratedGradients(
        lambda ids, mask: model(ids, mask),
        model.bert.embeddings.word_embeddings
    )

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_class,
        return_convergence_delta=True,
        n_steps=200
    )

    scores = attributions.sum(dim=-1).squeeze(0)
    scores = scores / scores.abs().max()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, scores.tolist(), delta.item()

tokens, scores, delta = explain("@USER lava tua boca suja, imunda, porca, nojenta prá falar do bolsonaro e a família. q moral você têm? vanpiro. link")

print("--- Raw tokens ---")
for token, score in zip(tokens, scores):
    print(f"{token:20s} {score:+.4f}")

print("\n--- Merged subwords ---")
for token, score in merge_subwords(tokens, scores):
    print(f"{token:20s} {score:+.4f}")

print(f"\nConvergence delta: {delta:.4f}")