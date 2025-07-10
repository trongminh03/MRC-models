from collections import Counter
import string
import re
import argparse
import json
import sys
from question_answering_fix import QuestionAnsweringModel, QuestionAnsweringArgs

def convert_data(data):
    converted = []
    for d in data:
        t = {}
        t['context'] = d['paragraphs'][0]['context']
        qas = []
        for q in d['paragraphs'][0]['qas']:
            q['is_impossible'] = False
            qas.append(q)
        t['qas'] = qas
        converted.append(t)
    return converted

def read_data(PATH):
    try:
        with open(PATH+'train.json', 'r', encoding='utf8') as f:
            TRAIN = json.load(f)
        with open(PATH+'dev.json', 'r', encoding='utf8') as f:
            TEST = json.load(f)
        return TRAIN, TEST
    except Exception as e:
        print(f"Error reading data: {e}")
        sys.exit(1)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + ' will receive score 0.'
                    print(message)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}

def load_model(type, path):
    try:
        model = QuestionAnsweringModel(type, path, args={"silent": False})
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/data")
    parser.add_argument('--type', type=str, default="auto")
    parser.add_argument('--output_path', type=str, default="/home/data")
    parser.add_argument('--is_test', default=False, action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser_args()
    print("Reading data...")
    train, test = read_data(args.path)

    print("Converting data...")
    train_squad = convert_data(train['data'])
    test_squad = convert_data(test['data'])

    print("Loading model...")
    model = load_model(args.type, args.output_path)

    if args.is_test:
        print('--test--')
        try:
            print("Running evaluation...")
            result, texts = model.eval_model(test_squad, output_dir=args.output_path, verbose=True)  # Lấy cả result và texts
            print("Evaluation result:", result)
        except Exception as e:
            print(f"Error during eval_model: {e}")
            sys.exit(1)

        try:
            with open(args.output_path + '/predictions_test.json', 'r', encoding='utf-8') as f:
                pred = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {args.output_path}/predictions_test.json not found.")
            sys.exit(1)

        # In kết quả tổng quát
        print("\nKết quả đánh giá tổng quát:")
        eval_result = evaluate(test['data'], pred)
        print(eval_result)

        # In tỷ lệ đúng và sai theo type_qas
        print("\nTỷ lệ đúng và sai theo loại câu hỏi (type_qas):")
        with open(args.output_path + '/type_qas_rates.txt', 'w', encoding='utf-8') as f:
            f.write("Tỷ lệ đúng và sai theo loại câu hỏi (type_qas):\n")
            if 'type_qas_stats' in result and result['type_qas_stats']:
                for type_qas, stats in result['type_qas_stats'].items():
                    correct_rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                    error_rate = stats['incorrect'] / stats['total'] if stats['total'] > 0 else 0.0
                    similar_rate = stats['similar'] / stats['total'] if stats['total'] > 0 else 0.0
                    print(f"{type_qas}: Đúng = {correct_rate:.4f} ({stats['correct']}/{stats['total']}), "
                          f"Gần đúng = {similar_rate:.4f} ({stats['similar']}/{stats['total']}), "
                          f"Sai = {error_rate:.4f} ({stats['incorrect']}/{stats['total']})")
                    f.write(f"{type_qas}: Đúng = {correct_rate:.4f} ({stats['correct']}/{stats['total']}), "
                             f"Gần đúng = {similar_rate:.4f} ({stats['similar']}/{stats['total']}), "
                             f"Sai = {error_rate:.4f} ({stats['incorrect']}/{stats['total']})\n")
            else:
                print("No type_qas_stats found or it is empty.")
                f.write("No type_qas_stats found or it is empty.\n")

        # In và lưu similar_text
        print("\nCác câu hỏi được phân loại là 'gần đúng' (similar_text):")
        with open(args.output_path + '/similar_text.json', 'w', encoding='utf-8') as f:
            json.dump(texts['similar_text'], f, ensure_ascii=False, indent=2)
        if texts['similar_text']:
            for q_id, info in texts['similar_text'].items():
                print(f"\nID: {q_id}")
                print(f"Câu hỏi: {info['question']}")
                print(f"Dự đoán: {info['predicted']}")
                print(f"Câu trả lời đúng: {info['truth']}")
        else:
            print("Không có câu hỏi nào được phân loại là 'gần đúng'.")