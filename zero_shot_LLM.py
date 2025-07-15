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
def build_prompt(context, question, few_shot_examples):
    prompt = "You are a question answering system. Extract the answer to each question using a span from the context.\n"
    
    for ex in few_shot_examples:
        prompt += f"\nExample:\nContext: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n"

    prompt += f"\nNow, answer the following:\nContext: {context}\nQuestion: {question}\nAnswer:"
    return prompt

# ---------------------- INFERENCE -------------------------- #
def generate_answer(model, tokenizer, examples, few_shot_examples):
    results = []
    for ex in tqdm(examples):
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
        results.append({"id": ex["id"], "prediction": answer})
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

    # Flatten test examples
    examples = []
    # test_data = test_data[:10]
    for item in test_data:
        for p in item["paragraphs"]:
            for q in p["qas"]:
                examples.append({
                    "context": re.sub("\s+", " ", p["context"]).strip(),
                    "question": q["question"].strip(),
                    "id": q["id"],
                    "answers": [a["text"] for a in q["answers"]]
                })

    print(f"Loaded {len(examples)} test examples.")

    # Run inference
    all_results = generate_answer(model, tokenizer, examples, few_shot_examples)

    # Collect predictions
    predictions = {res["id"]: res["prediction"] for res in all_results}

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

    print(f"\nFew-shot LLM | EM: {em_score:.2f} | F1: {f1_score_final:.2f}")

    with open(args.output_path, "w", encoding="utf8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions to {args.output_path}")


# import re
# import json
# import argparse
# from tqdm import tqdm
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from collections import Counter
# import string

# # --------------------------- CONFIG ---------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, required=True)
# parser.add_argument("--test_path", type=str, required=True)
# parser.add_argument("--output_path", type=str, default="visqa_llm_results.json")
# parser.add_argument("--max_new_tokens", type=int, default=50)
# args = parser.parse_args()

# # ---------------------- EVAL METRICS --------------------------
# def normalize_answer(s):
#     def white_space_fix(text): return ' '.join(text.split())
#     def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
#     def lower(text): return text.lower()
#     return white_space_fix(remove_punc(lower(s)))

# def f1_score(pred, truth):
#     pred_tokens = normalize_answer(pred).split()
#     truth_tokens = normalize_answer(truth).split()
#     common = Counter(pred_tokens) & Counter(truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0: return 0
#     precision = num_same / len(pred_tokens)
#     recall = num_same / len(truth_tokens)
#     return (2 * precision * recall) / (precision + recall)

# def exact_match(pred, truth):
#     return normalize_answer(pred) == normalize_answer(truth)

# # ---------------------- PROMPT --------------------------
# def build_prompt(context, question):
# #     return f"""Trích xuất câu trả lời chính xác từ đoạn văn sau để trả lời câu hỏi.

# # Đoạn văn:
# # \"{context}\"

# # Câu hỏi:
# # \"{question}\"

# # Chỉ trích xuất cụm từ hoặc câu nằm trong đoạn văn làm câu trả lời:"""
#     prompt = f"""You are a question answering system. Extract the answer to each question using a span from the context.

#                 Example 1:
#                 Context: phạm văn đồng, một tháng ba năm một nghìn chín trăm lẻ sáu đến hai mươi chín tháng bốn năm hai nghìn, là thủ tướng đầu tiên của nước cộng hòa xã hội chủ nghĩa việt nam từ năm một nghìn chín trăm bảy mươi sáu, từ năm một nghìn chín trăm tám mươi mốt, gọi là chủ tịch hội đồng bộ trưởng, cho đến khi nghỉ hưu năm một nghìn chín trăm tám mươi bảy. trước đó, ông từng giữ chức vụ thủ tướng chính phủ việt nam dân chủ cộng hòa từ năm một nghìn chín trăm năm mươi lăm đến năm một nghìn chín trăm bảy mươi sáu. ông là vị thủ tướng việt nam tại vị lâu nhất, năm một nghìn chín trăm năm mươi lăm đến năm một nghìn chín trăm tám mươi bảy. ông là học trò, cộng sự của chủ tịch hồ chí minh. ông có tên gọi thân mật là tô, đây từng là bí danh của ông. ông còn có tên gọi là lâm bá kiệt, khi làm phó chủ nhiệm cơ quan biện sự sứ tại quế lâm, chủ nhiệm là hồ học lãm..
#                 Question: tên gọi nào được phạm văn đồng sử dụng khi làm phó chủ nhiệm cơ quan biện sự xứ tại quế lâm?
#                 Answer: lâm bá kiệt

#                 Example 2:
#                 Context: sự phát hiện của hofmeister năm một nghìn tám trăm năm mươi mốt về các thay đổi xảy ra trong túi phôi của thực vật có hoa, cũng như sự xác định của ông về các quan hệ chính xác của các thay đổi này với thực vật có mạch, đã cố định vị trí của gymnosperm như là một lớp phân biệt với thực vật hai lá mầm, jelly và thuật ngữ angiosperm, sau đó dần dần được chấp nhận như là tên gọi phù hợp cho toàn bộ thực vật có hoa hơn là gymnosperm, và nó bao gồm trong đó các lớp thực vật hai lá mầm và thực vật một lá mầm. đây chính là ý nghĩa mà thuật ngữ này hiện nay có được và được sử dụng ở đây.
#                 Question: năm một nghìn tám trăm năm mươi mốt nhà sinh học hofmeister đã tìm ra điều gì ở thực vật có hoa?
#                 Answer: các thay đổi xảy ra trong túi phôi của thực vật có hoa

#                 Now, answer the following:
#                 Context: {context}
#                 Question: {question}
#                 Answer:"""
#     return prompt

# # ---------------------- PROCESS --------------------------
# def generate_answer(model, tokenizer, examples):
#     results = []
#     for ex in tqdm(examples):
#         prompt = build_prompt(ex["context"], ex["question"])
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
#         answer = decoded[len(prompt):].strip()
#         results.append({"id": ex["id"], "prediction": answer})
#     return results

# # ---------------------- MAIN --------------------------
# if __name__ == "__main__":
#     # Load tokenizer + model with model parallelism across 2 GPUs
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name,
#         device_map="auto",  # automatically split across multiple GPUs
#         torch_dtype=torch.bfloat16,  # or torch.float16 if bfloat16 unsupported
#         trust_remote_code=True
#     )
#     model.eval()

#     # Load ViSQA test data
#     with open(args.test_path, encoding="utf8") as f:
#         test_data = json.load(f)["data"]

#     # Flatten examples
#     examples = []
    
#     # get only 10 samples for testing
#     test_data = test_data[:10]
#     for item in test_data:
#         for p in item["paragraphs"]:
#             for q in p["qas"]:
#                 examples.append({
#                     "context": re.sub("\s+", " ", p["context"]).strip(),
#                     "question": q["question"].strip(),
#                     "id": q["id"],
#                     "answers": [a["text"] for a in q["answers"]]
#                 })

    
#     # import IPython; IPython.embed()
#     print(f"Loaded {len(examples)} examples.")

#     # Run sequential inference (safe for multi-GPU)
#     all_results = generate_answer(model, tokenizer, examples)

#     # Collect predictions
#     predictions = {res["id"]: res["prediction"] for res in all_results}

#     # Evaluate
#     em_total, f1_total = 0, 0
#     for ex in examples:
#         pid = ex["id"]
#         pred = predictions[pid]
#         # import IPython; IPython.embed()  # Debugging line
#         em = max(exact_match(pred, ans) for ans in ex["answers"])
#         f1 = max(f1_score(pred, ans) for ans in ex["answers"])
#         em_total += em
#         f1_total += f1

#     em_score = 100 * em_total / len(examples)
#     f1_score_final = 100 * f1_total / len(examples)

#     print(f"Zero-Shot LLM | EM: {em_score:.2f} | F1: {f1_score_final:.2f}")

#     # Save results
#     with open(args.output_path, "w", encoding="utf8") as f:
#         json.dump(predictions, f, indent=2, ensure_ascii=False)
#     print(f"Saved predictions to {args.output_path}")
