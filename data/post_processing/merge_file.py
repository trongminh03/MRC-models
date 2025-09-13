import json
import os
import argparse

def merge_jsonl_files(questions_file, context_file, output_file):
    # Read the questions file
    questions_data = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                questions_data.append(data)
    
    # Create a map of context_id to context
    context = {}
    with open(context_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'context_id' in data and 'context' in data:
                    context[data['context_id']] = data['context']
    
    # Add context to questions
    for question in questions_data:
        if 'context_id' in question and question['context_id'] in context:
            question['context'] = context[question['context_id']]
    
    # Write the updated question data to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in questions_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Count how many questions have context
    questions_with_context = sum(1 for q in questions_data if 'context' in q)
    
    print(f"Processed {len(questions_data)} questions.")
    print(f"Added context to {questions_with_context} questions.")
    print(f"Output saved to {output_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Add context to question JSONL files by context_id')
    parser.add_argument('-q', '--question', type=str, required=True,
                        help='Path to the questions JSONL file')
    parser.add_argument('-c', '--context', type=str, required=True,
                        help='Path to the context JSONL file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to the output JSONL file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    merge_jsonl_files(args.question, args.context, args.output)