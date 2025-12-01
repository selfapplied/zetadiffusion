# Script Standards

All Python scripts in this project follow these conventions:

## Hashbang

All executable scripts use:
```python
#!.venv/bin/python
```

This ensures scripts run with the project's virtual environment.

## Executability

All scripts are made executable:
```bash
chmod +x script_name.py
```

This allows direct execution:
```bash
./script_name.py
```

Instead of:
```bash
python3 script_name.py
```

## Updated Scripts

The following scripts have been updated to follow these standards:

- `setup_notion.py` - Notion API setup
- `save_analysis_results.py` - Save analysis results
- `upload_to_notion.py` - Upload to Notion
- `demo_operator_analysis.py` - Operator analysis demo
- `example_notion_integration.py` - Integration examples
- `demo_bert_compression.py`
- `demo_interactive_visualization.py`
- `validate_conjecture_9_1_1.py`
- `validate_temperature_cascade.py`
- `demo_crypto_predictions.py`
- `demo_wormhole_markets.py`
- `demo_universal_fields.py`
- `generate_from_bert.py`
- `demo_energy_harvesting.py`
- `demo_compression_showcase.py`
- `demo.py`

## Usage

Scripts can now be run directly:
```bash
./setup_notion.py
./save_analysis_results.py
./upload_to_notion.py
./demo_operator_analysis.py
```

## Virtual Environment

The hashbang assumes `.venv` exists. To create it:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Gitignore

The following are ignored:
- `.venv/` - Virtual environment
- `.env` - Environment variables (API tokens, etc.)
- `.out/` - Output files

