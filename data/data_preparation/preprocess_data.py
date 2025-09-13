#!/usr/bin/env python

"""
Replace Contexts with Context IDs
--------------------------------
Reads a JSONL file and replaces the full context text with context IDs
from a previously generated distinct contexts file.
"""

import os
import json
import argparse

def read_jsonl(file_path):
    """Read data from JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        print(f"Read {len(data)} records from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading JSONL file {file_path}: {e}")
        return []

def read_json(file_path):
    """Read data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the data is nested under 'contexts'
        if 'contexts' in data and isinstance(data['contexts'], list):
            print(f"Read {len(data['contexts'])} contexts from JSON file")
            return data['contexts']
        else:
            print(f"Read data from JSON file")
            return data
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return []

def build_context_mapping(contexts_data):
    """Build a mapping from context text to context ID."""
    context_map = {}
    
    for item in contexts_data:
        if 'context' in item and 'context_id' in item:
            context_map[item['context']] = item['context_id']
    
    print(f"Built mapping for {len(context_map)} distinct contexts")
    return context_map

def replace_contexts(data, context_map):
    """Replace full contexts with context IDs."""
    modified_data = []
    replaced_count = 0
    missed_count = 0
    
    for item in data:
        modified_item = item.copy()
        
        if 'context' in modified_item and modified_item['context'].strip():
            context_text = modified_item['context'].strip()
            
            if context_text in context_map:
                # Replace context with context_id
                modified_item['context_id'] = context_map[context_text]
                del modified_item['context']  # Remove original context to save space
                replaced_count += 1
            else:
                # If context not found in mapping, keep it as is but log warning
                print(f"Warning: Context not found in mapping: '{context_text[:50]}...'")
                missed_count += 1
        
        modified_data.append(modified_item)
    
    print(f"Replaced {replaced_count} contexts, missed {missed_count} contexts")
    return modified_data

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Replace contexts with context IDs')
    parser.add_argument('--input', required=True,
                      help='Input JSONL file with full contexts')
    parser.add_argument('--contexts', required=True,
                      help='File containing distinct contexts mapping (JSONL or JSON)')
    parser.add_argument('--output', required=True,
                      help='Output JSONL file with context IDs')
    
    args = parser.parse_args()
    
    # Determine contexts file format and read it
    if args.contexts.lower().endswith('.json'):
        contexts_data = read_json(args.contexts)
    else:
        contexts_data = read_jsonl(args.contexts)
    
    if not contexts_data:
        print("No context mapping data. Exiting...")
        return
    
    # Build context to ID mapping
    context_map = build_context_mapping(contexts_data)
    
    # Read input data
    data = read_jsonl(args.input)
    if not data:
        print("No input data to process. Exiting...")
        return
    
    # Replace contexts with IDs
    modified_data = replace_contexts(data, context_map)
    
    # Write modified data
    write_jsonl(modified_data, args.output)
    
    print("Replacement complete!")

if __name__ == "__main__":
    main() 