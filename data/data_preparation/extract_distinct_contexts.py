#!/usr/bin/env python

"""
Extract Distinct Contexts
------------------------
Reads a JSONL or JSON file containing UIT-ViQuAD data, extracts distinct contexts,
and writes a JSONL file mapping context IDs to their full text.
"""

import os
import json
import hashlib
import argparse

def read_data(file_path):
    """Read data from JSONL or JSON file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.lower().endswith('.json'):
                # Handle regular JSON file
                json_data = json.load(f)
                
                # Check if it's an array or a dictionary with data in a field
                if isinstance(json_data, list):
                    data = json_data
                elif isinstance(json_data, dict):
                    # Look for data in common fields (adapt as needed)
                    for field in ['data', 'examples', 'records', 'items', 'train', 'validation', 'test']:
                        if field in json_data and isinstance(json_data[field], list):
                            data.extend(json_data[field])
                            # break
                    
                    # If no array found in common fields, use the dict itself
                    if not data:
                        data = [json_data]
            else:
                # Handle JSONL file (default)
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
                        
        print(f"Read {len(data)} records from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def extract_distinct_contexts(data):
    """Extract distinct contexts from the dataset and assign IDs."""
    contexts = {}  # Maps context hash to context info
    
    # Find unique contexts
    for idx, item in enumerate(data):
        if 'context' in item and item['context'].strip():
            context = item['context'].strip()
            context_hash = hashlib.md5(context.encode('utf-8')).hexdigest()
            
            if context_hash not in contexts:
                # Generate a context ID based on hash
                context_id = f"ctx_{context_hash[:8]}"
                
                contexts[context_hash] = {
                    'context_id': context_id,
                    'context': context
                }
    
    # Convert to list
    contexts_list = list(contexts.values())
    print(f"Extracted {len(contexts_list)} distinct contexts")
    return contexts_list

def write_jsonl(data, file_path):
    """Write data to JSONL file."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Wrote {len(data)} records to {file_path}")
        return True
    except Exception as e:
        print(f"Error writing JSONL file {file_path}: {e}")
        return False

def write_json(data, file_path):
    """Write data to a JSON file."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'contexts': data}, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(data)} records to {file_path}")
        return True
    except Exception as e:
        print(f"Error writing JSON file {file_path}: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract distinct contexts from JSONL or JSON file')
    parser.add_argument('--input', default=os.path.join('data', 'raw', 'jsonl', 'uit-viquad_train.jsonl'),
                      help='Input file (JSONL or JSON format)')
    parser.add_argument('--output', default=os.path.join('data', 'processed', 'contexts.jsonl'),
                      help='Output file (default: data/processed/contexts.jsonl)')
    parser.add_argument('--format', choices=['jsonl', 'json'], default='jsonl',
                      help='Output format (jsonl or json, default: jsonl)')
    
    args = parser.parse_args()
    
    # Read input data
    data = read_data(args.input)
    if not data:
        print("No data to process. Exiting...")
        return
    
    # Extract distinct contexts
    distinct_contexts = extract_distinct_contexts(data)
    
    # Write distinct contexts to output file in requested format
    if args.format == 'json':
        output_path = args.output
        if not output_path.lower().endswith('.json'):
            output_path = output_path.rsplit('.', 1)[0] + '.json'
        write_json(distinct_contexts, output_path)
    else:
        write_jsonl(distinct_contexts, args.output)
    
    print("Extraction complete!")

if __name__ == "__main__":
    main() 