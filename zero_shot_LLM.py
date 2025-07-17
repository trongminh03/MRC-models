import re
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter, defaultdict
import string
import random

# --------------------------- CONFIG --------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--test_path", type=str, required=True)
parser.add_argument("--train_path", type=str, required=True)
parser.add_argument("--output_path", type=str, default="visqa_llm_results.json")
parser.add_argument("--max_new_tokens", type=int, default=50)
parser.add_argument("--n_few_shot", type=int, default=2)
args = parser.parse_args()

# ---------------------- EVAL METRICS -------------------------- #
def normalize_answer(s):
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def f1_score(pred, truth):
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match(pred, truth):
    return normalize_answer(pred) == normalize_answer(truth)

# ---------------------- LOAD FEW-SHOT EXAMPLES -------------------------- #
def load_few_shot_examples(train_data, n):
    seen_contexts = set()
    candidates = []

    for item in train_data:
        for para in item["paragraphs"]:
            context = re.sub(r"\s+", " ", para["context"]).strip()
            if context in seen_contexts:
                continue
            seen_contexts.add(context)
            for qa in para["qas"]:
                if len(qa["answers"]) > 0:
                    candidates.append({
                        "context": context,
                        "question": qa["question"].strip(),
                        "answer": qa["answers"][0]["text"].strip()
                    })
                break  # Use only 1 QA per unique context

    return random.sample(candidates, min(n, len(candidates)))

# ---------------------- PROMPT -------------------------- #
# def build_prompt(context, question, few_shot_examples):
#     prompt = "You are a question answering system. Extract the answer to each question using a span from the context.\n"
    
#     for ex in few_shot_examples:
#         prompt += f"\nExample:\nContext: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n"

#     prompt += f"\nNow, answer the following:\nContext: {context}\nQuestion: {question}\nAnswer:"
#     return prompt

def build_prompt(context, question, few_shot_examples):
    prompt = (
        "Answer each question by copying a phrase or sentence exactly from the context.\n"
        "Do not guess or rephrase. Only extract what's in the text.\n"
    )

    for ex in few_shot_examples:
        prompt += f"\nContext: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n"

    prompt += (
        f"\nContext: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt

# ---------------------- INFERENCE -------------------------- #
# def generate_answer(model, tokenizer, examples, few_shot_examples):
#     results = []
#     for ex in tqdm(examples):
#         prompt = build_prompt(ex["context"], ex["question"], few_shot_examples)
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=args.max_new_tokens,
#                 do_sample=False,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#         decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         answer = decoded[len(prompt):].strip().split("\n")[0]
#         results.append({"id": ex["id"], "prediction": answer})
#     return results

def generate_answer(model, tokenizer, examples, few_shot_examples, output_path):
    results = []

    # Load existing results if output file exists
    existing_results = []
    existing_ids = set()

    try:
        with open(output_path, "r", encoding="utf8") as f:
            existing_results = json.load(f)
            existing_ids = {res["id"] for res in existing_results}
            print(f"Found {len(existing_ids)} existing predictions. Will append remaining.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing output or corrupted file. Starting fresh.")

    with open(output_path, "w", encoding="utf8") as f:
        f.write("[\n")  # Start JSON list

        # Write existing results first
        for i, res in enumerate(existing_results):
            json.dump(res, f, ensure_ascii=False)
            f.write(",\n")

        for i, ex in enumerate(tqdm(examples)):
            if ex["id"] in existing_ids:
                continue  # Already processed
            
            prompt = build_prompt(ex["context"], ex["question"], few_shot_examples)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = decoded[len(prompt):].strip().split("\n")[0]
            result = {"id": ex["id"], "prediction": answer}
            results.append(result)

            # Write prediction to file immediately
            json.dump(result, f, ensure_ascii=False)
            if i < len(examples) - 1:
                f.write(",\n")
            else:
                f.write("\n")

             # 🧹 Explicit memory cleanup
            del inputs, outputs, decoded, answer
            torch.cuda.empty_cache()

        f.write("]\n")  # End of JSON list
    return results

# ---------------------- MAIN -------------------------- #
if __name__ == "__main__":
    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    # Load test set
    with open(args.test_path, encoding="utf8") as f:
        test_data = json.load(f)["data"]

    # Load training set for few-shot
    with open(args.train_path, encoding="utf8") as f:
        train_data = json.load(f)["data"]

    few_shot_examples = load_few_shot_examples(train_data, args.n_few_shot)
    print(f"Loaded {len(few_shot_examples)} few-shot examples.")

    # Load already processed IDs from output_path
    already_processed_ids = set()
    try:
        with open(args.output_path, "r", encoding="utf8") as f:
            partial_results = json.load(f)
            for item in partial_results:
                already_processed_ids.add(item["id"])
            print(f"Resuming from checkpoint. Found {len(already_processed_ids)} already processed predictions.")
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        print(f"No existing results found at {args.output_path}. Starting fresh.")

    # Flatten test examples
    examples = []
    # test_data = test_data[:10]
    for item in test_data:
        for p in item["paragraphs"]:
            for q in p["qas"]:
                if q["id"] in already_processed_ids:
                    continue    
                examples.append({
                    "context": re.sub("\s+", " ", p["context"]).strip(),
                    "question": q["question"].strip(),
                    "id": q["id"],
                    "answers": [a["text"] for a in q["answers"]]
                })

    print(f"Loaded {len(examples)} test examples.")

    # Run inference
    all_results = generate_answer(model, tokenizer, examples, few_shot_examples, output_path=args.output_path)

    # Collect predictions
    predictions = {res["id"]: res["prediction"] for res in all_results}

    # Evaluate
    em_total, f1_total = 0, 0

    # examples = examples[:5]
    for ex in examples:
        pid = ex["id"]
        pred = predictions[pid]
        # import IPython; IPython.embed()  # Debugging line
        em = max(exact_match(pred, ans) for ans in ex["answers"])
        f1 = max(f1_score(pred, ans) for ans in ex["answers"])
        em_total += em
        f1_total += f1

    em_score = 100 * em_total / len(examples)
    f1_score_final = 100 * f1_total / len(examples)

    print(f"\nFew-shot LLM | EM: {em_score:.2f} | F1: {f1_score_final:.2f}")

    # with open(args.output_path, "w", encoding="utf8") as f:
    #     json.dump(predictions, f, indent=2, ensure_ascii=False)
    # print(f"Saved predictions to {args.output_path}")
