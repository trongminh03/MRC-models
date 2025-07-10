import re
import json
import argparse
from tqdm import tqdm
from pqdm.processes import pqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer

from collections import Counter
import string

# --------------------------- CONFIG ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--test_path", type=str, required=True)
parser.add_argument("--output_path", type=str, default="visqa_llm_results.json")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--max_new_tokens", type=int, default=50)
args = parser.parse_args()

# ---------------------- EVAL METRICS --------------------------
def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def f1_score(pred, truth):
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match(pred, truth):
    return normalize_answer(pred) == normalize_answer(truth)

# ---------------------- PROMPT --------------------------
def build_prompt(context, question):
    return f"""Hãy trả lời câu hỏi dựa trên đoạn văn bên dưới.

Đoạn văn:
\"{context}\"

Câu hỏi:
\"{question}\"

Câu trả lời ngắn:"""

# ---------------------- PROCESS --------------------------
def generate_answer(examples):
    results = []
    local_tokenizer = local_tokenizer_ref
    local_model = local_model_ref

    for ex in tqdm(examples):
        prompt = build_prompt(ex["context"], ex["question"])
        inputs = local_tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = local_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=local_tokenizer.eos_token_id
            )
        decoded = local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded[len(prompt):].strip()
        results.append({"id": ex["id"], "prediction": answer})
    return results

# ---------------------- MAIN --------------------------
if __name__ == "__main__":
    torch.set_default_device("cuda")

    # Load tokenizer + model
    # if "llama" in args.model_name.lower():
    #     tokenizer = LlamaTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    #     model = LlamaForCausalLM.from_pretrained(
    #         args.model_name,
    #         device_map="cuda",
    #         torch_dtype=torch.bfloat16,
    #         trust_remote_code=True
    #     )
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model_name,
    #         device_map="cuda",
    #         torch_dtype=torch.bfloat16,
    #         trust_remote_code=True
    #     )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True)

    local_tokenizer_ref = tokenizer
    local_model_ref = model

    # Load ViSQA test data
    with open(args.test_path, encoding="utf8") as f:
        test_data = json.load(f)["data"]

    # Flatten examples
    examples = []
    for item in test_data:
        for p in item["paragraphs"]:
            for q in p["qas"]:
                examples.append({
                    "context": re.sub("\s+", " ", p["context"]).strip(),
                    "question": q["question"].strip(),
                    "id": q["id"],
                    "answers": [a["text"] for a in q["answers"]]
                })

    print(f"Loaded {len(examples)} examples.")

    # Split into batches for threads
    batch_size = len(examples) // args.threads + 1
    chunks = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]

    # Parallel generation
    all_results = pqdm(chunks, generate_answer, n_jobs=args.threads)

    # Flatten results
    predictions = {}
    for batch in all_results:
        for res in batch:
            predictions[res["id"]] = res["prediction"]

    # Evaluate
    em_total, f1_total = 0, 0
    for ex in examples:
        pid = ex["id"]
        pred = predictions[pid]
        em = max(exact_match(pred, ans) for ans in ex["answers"])
        f1 = max(f1_score(pred, ans) for ans in ex["answers"])
        em_total += em
        f1_total += f1

    em_score = 100 * em_total / len(examples)
    f1_score_final = 100 * f1_total / len(examples)

    print(f"Zero-Shot LLM | EM: {em_score:.2f} | F1: {f1_score_final:.2f}")

    # Save
    with open(args.output_path, "w", encoding="utf8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions to {args.output_path}")
