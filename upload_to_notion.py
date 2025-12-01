#!.venv/bin/python
"""
upload_to_notion.py

Uploads validation results to Notion database.
Now uses shared validation framework - uploads happen automatically.
This script is for manual re-uploads if needed.
"""

import os
import json
from pathlib import Path
from notion_logger import ValidationRunLogger

def upload_all_results():
    """Upload all available validation results (legacy support)."""
    
    logger = ValidationRunLogger()
    uploaded = []
    
    # Find all result files
    result_files = list(Path(".out").glob("*_results.json"))
    
    for result_file in result_files:
        print(f"Uploading {result_file.name}...")
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract validation type from filename or data
            validation_type = data.get("validation_type", result_file.stem.replace("_results", "").replace("_", " ").title())
            parameters = data.get("parameters", {})
            results = data.get("results", data)
            execution_time = data.get("execution_time", 0)
            
            page_id = logger.create_run_entry(
                validation_type=validation_type,
                parameters=parameters,
                results=results,
                execution_time=execution_time,
                output_files=[str(result_file)]
            )
            if page_id:
                uploaded.append((validation_type, page_id))
        except Exception as e:
            print(f"  ⚠ Error uploading {result_file.name}: {e}")
    
    print(f"\n✓ Uploaded {len(uploaded)} validation runs to Notion")
    return uploaded

if __name__ == "__main__":
    print("=" * 70)
    print("Uploading Validation Results to Notion")
    print("=" * 70)
    print("\nNote: New validations auto-upload via validation_framework.")
    print("This script is for manual re-uploads.\n")
    
    if not os.getenv("NOTION_API_TOKEN"):
        print("⚠ NOTION_API_TOKEN not set. Skipping upload.")
        print("Set it with: export NOTION_API_TOKEN='your_token'")
    else:
        uploaded = upload_all_results()
        if uploaded:
            print("\nUploaded runs:")
            for name, page_id in uploaded:
                print(f"  - {name}: {page_id}")

