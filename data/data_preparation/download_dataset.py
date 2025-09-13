from datasets import load_dataset
import json
import os

# Load the dataset
print("Loading dataset...")
ds = load_dataset("taidng/UIT-ViQuAD2.0")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Process and save each split
for split in ["train", "validation", "test"]:
    print(f"Processing {split} split...")
    split_data = ds[split]
    
    # Prepare filename
    output_path = f"data/uit-viquad_{split}.jsonl"
    
    # Save as JSONL (JSON Lines)
    with open(output_path, "w", encoding="utf-8") as f:
        if len(split_data) > 0:
            # Write each example as a separate JSON line
            for example in split_data:
                # Convert to dict and write as a JSON line
                json_line = json.dumps(dict(example), ensure_ascii=False)
                f.write(json_line + "\n")
                
            print(f"Saved {len(split_data)} examples to {output_path}")
        else:
            print(f"No data in {split} split")

print("All splits saved as JSONL files in the data directory")