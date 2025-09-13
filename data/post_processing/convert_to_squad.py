#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
from collections import defaultdict

def convert_jsonl_to_squad(input_file, output_file, version="v1.0"):
    """
    Convert JSONL format to SQuAD-style JSON format.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSON file
        version: Version string for the output format
    """
    # Read the JSONL file
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    # Group by title and context
    title_context_map = defaultdict(lambda: defaultdict(list))
    for item in items:
        title = item.get('title', 'Untitled')
        context = item.get('context', '')
        
        # Skip items without context
        if not context:
            continue
            
        # Add this question to the appropriate title and context group
        title_context_map[title][context].append(item)
    
    # Build the SQuAD format
    squad_data = {
        "version": version,
        "data": []
    }
    
    # Process each title
    for title, contexts in title_context_map.items():
        title_entry = {
            "title": title,
            "paragraphs": []
        }
        
        # Process each context under this title
        for context, questions in contexts.items():
            paragraph = {
                "context": context,
                "qas": []
            }
            
            # Process each question for this context
            for question in questions:
                qa_item = {
                    "id": question.get('id', question.get('uit_id', 'unknown')),
                    "question": question.get('question', ''),
                    "answers": []
                }
                
                # Process answers
                if 'answers' in question and question['answers'] and isinstance(question['answers'], dict):
                    answer_texts = question['answers'].get('text', [])
                    answer_starts = question['answers'].get('answer_start', [])
                    
                    # Ensure we have matching text and start positions
                    if len(answer_texts) == len(answer_starts):
                        for i in range(len(answer_texts)):
                            if answer_texts[i]:  # Skip empty answers
                                qa_item['answers'].append({
                                    "text": answer_texts[i],
                                    "answer_start": answer_starts[i]
                                })
                
                # Only add questions that have answers
                if qa_item['answers']:
                    paragraph['qas'].append(qa_item)
            
            # Only add paragraphs that have questions
            if paragraph['qas']:
                title_entry['paragraphs'].append(paragraph)
        
        # Only add titles that have paragraphs
        if title_entry['paragraphs']:
            squad_data['data'].append(title_entry)
    
    # Write the output JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(squad_data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    num_titles = len(squad_data['data'])
    num_paragraphs = sum(len(entry['paragraphs']) for entry in squad_data['data'])
    num_questions = sum(sum(len(para['qas']) for para in entry['paragraphs']) for entry in squad_data['data'])
    
    print(f"Conversion complete.")
    print(f"Created {num_titles} titles, {num_paragraphs} paragraphs, and {num_questions} questions.")
    print(f"Output saved to: {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert JSONL to SQuAD-style JSON format")
    parser.add_argument("--input", help="Path to the input JSONL file")
    parser.add_argument("--output", help="Path to the output JSON file")
    parser.add_argument("--version", default="v1.0", help="Version string for the output format")
    args = parser.parse_args()
    
    # Set default output file path if not provided
    if not args.output:
        input_base = os.path.splitext(args.input)[0]
        args.output = f"{input_base}_squad.json"
    
    # Process the file
    convert_jsonl_to_squad(args.input, args.output, args.version)

if __name__ == "__main__":
    main() 