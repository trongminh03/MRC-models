#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import argparse
from tqdm import tqdm
from normalize import normalize_vietnamese_text

def get_nested_value(obj, path):
    """
    Get a value from a nested dictionary using a dot-separated path.
    
    Args:
        obj: Dictionary to navigate
        path: Dot-separated string path (e.g., 'answers.text')
    
    Returns:
        The value at the specified path or None if the path doesn't exist
    """
    parts = path.split('.')
    current = obj
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    
    return current

def set_nested_value(obj, path, value):
    """
    Set a value in a nested dictionary using a dot-separated path.
    
    Args:
        obj: Dictionary to navigate and modify
        path: Dot-separated string path (e.g., 'answers.text')
        value: Value to set at the specified path
    
    Returns:
        True if the value was set, False otherwise
    """
    parts = path.split('.')
    current = obj
    
    # Navigate to the parent of the leaf node
    for i, part in enumerate(parts[:-1]):
        if isinstance(current, dict):
            # If the part doesn't exist or isn't a dict, create it
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        else:
            return False
    
    # Set the value at the leaf node
    if isinstance(current, dict):
        current[parts[-1]] = value
        return True
    
    return False

def normalize_jsonl_file(input_file, output_file, field_to_normalize='context'):
    """
    Normalize Vietnamese text in a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL file
        field_to_normalize: The field in each JSON object to normalize (default: 'context')
                           Can use dot notation for nested fields (e.g., 'answers.text')
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Count lines in the file for progress bar
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Process each line
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, total=total_lines, desc="Normalizing JSONL"):
            # Skip empty lines
            if not line.strip():
                continue
            
            try:
                # Parse JSON object
                json_obj = json.loads(line)
                
                # Check if we're using a simple field or a nested path
                if '.' in field_to_normalize:
                    # Get the nested value
                    value = get_nested_value(json_obj, field_to_normalize)
                    if isinstance(value, str):
                        # Normalize and set the value back
                        normalized_value = normalize_vietnamese_text(value)
                        set_nested_value(json_obj, field_to_normalize, normalized_value)
                    elif isinstance(value, list):
                        # Handle array of strings
                        normalized_array = []
                        for item in value:
                            if isinstance(item, str):
                                normalized_array.append(normalize_vietnamese_text(item))
                            else:
                                normalized_array.append(item)
                        set_nested_value(json_obj, field_to_normalize, normalized_array)
                elif field_to_normalize in json_obj:
                    if isinstance(json_obj[field_to_normalize], str):
                        # Normalize simple field with string
                        json_obj[field_to_normalize] = normalize_vietnamese_text(json_obj[field_to_normalize])
                    elif isinstance(json_obj[field_to_normalize], list):
                        # Normalize simple field with array of strings
                        normalized_array = []
                        for item in json_obj[field_to_normalize]:
                            if isinstance(item, str):
                                normalized_array.append(normalize_vietnamese_text(item))
                            else:
                                normalized_array.append(item)
                        json_obj[field_to_normalize] = normalized_array
                
                # Write the normalized JSON object to the output file
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:100]}...")
            except Exception as e:
                print(f"Error processing line: {str(e)}")
    
    print(f"Normalization complete. Output saved to: {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Normalize Vietnamese text in JSONL files")
    parser.add_argument("--input", help="Path to the input JSONL file")
    parser.add_argument("--output", help="Path to the output JSONL file")
    parser.add_argument("--field", default="context", 
                        help="JSON field to normalize (default: 'context'). Can use dot notation for nested fields (e.g., 'answers.text')")
    args = parser.parse_args()
    
    # Set default output file path if not provided
    if not args.output:
        input_base = os.path.splitext(args.input)[0]
        args.output = f"{input_base}_normalized.jsonl"
    
    # Process the file
    normalize_jsonl_file(args.input, args.output, args.field)

if __name__ == "__main__":
    main() 