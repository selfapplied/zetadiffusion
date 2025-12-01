# Notion Upload Standard Practice

**Effective:** December 1, 2025  
**Status:** ✅ Implemented

---

## Standard Practice

**All validation runs now automatically upload:**
1. ✅ **Results JSON** - Validation data and metrics
2. ✅ **Analysis Reports** - Markdown findings documents
3. ✅ **Diagnostic Plots** - PNG visualizations

**No manual steps required** - everything happens automatically when using `validation_framework.run_validation()`.

---

## How It Works

### Automatic File Discovery

When a validation runs, the system automatically:

1. **Finds related reports:**
   - Searches for `*FINDINGS*.md`, `*ANALYSIS*.md`, `*SUMMARY*.md`
   - Matches by validation type keywords (e.g., "conjecture", "feg", "clock")

2. **Finds related plots:**
   - Searches `.out/plots/` directory
   - Matches by validation type keywords
   - Includes all relevant PNG files

3. **Uploads to Notion:**
   - Reports → Converted to Notion blocks (headings, paragraphs, code blocks)
   - Plots → Added as callout blocks with file references

### Example

When running `validate_conjecture_9_1_3.py`:

**Automatically finds:**
- `CONJECTURE_9_1_3_FINDINGS.md` → Uploaded as report content
- `plot12_conjecture_9_1_3_three_clocks.png` → Added as plot reference
- `plot11_conjecture_9_1_2_bifurcation.png` → Added as plot reference
- Related conjecture plots → All added

**Result:** Complete Notion page with:
- Validation results
- Analysis report (full markdown content)
- Plot references (with file names and sizes)

---

## File Naming Conventions

### Reports
- `*FINDINGS*.md` - Detailed analysis findings
- `*ANALYSIS*.md` - Analysis documents
- `*SUMMARY*.md` - Summary documents

**Examples:**
- `CONJECTURE_9_1_3_FINDINGS.md`
- `CLOCK_EXECUTION_TIME_FINDINGS.md`
- `RESEARCH_FINDINGS.md`

### Plots
- Stored in `.out/plots/`
- Named with validation type keywords:
  - `*conjecture*.png`
  - `*feg*.png`
  - `*clock*.png`
  - `*temperature*.png`
  - `*operator*.png`

**Examples:**
- `plot12_conjecture_9_1_3_three_clocks.png`
- `plot13_clock_execution_times.png`
- `plot1_feg_coherence_vs_chaos.png`

---

## Implementation Details

### Markdown Conversion

Reports are converted to Notion blocks:
- `# Heading` → Heading 1
- `## Heading` → Heading 2
- `### Heading` → Heading 3
- `**bold**` → Bold text
- `*italic*` → Italic text
- `` `code` `` → Code blocks
- `- List` → Bulleted lists
- `[link](url)` → Links

### Plot Handling

**Current:** Plots are added as callout blocks with file references:
```
📊 Plot: plot12_conjecture_9_1_3_three_clocks.png (184 KB)
```

**Future Enhancement:** 
- Upload plots to external hosting (imgur, S3, etc.)
- Embed actual images in Notion pages
- Automatic image upload on validation run

---

## Usage

### Standard Validation

```python
from zetadiffusion.validation_framework import run_validation

def my_validation():
    # ... validation code ...
    return results

result = run_validation(
    validation_type="My Validation",
    validation_func=my_validation,
    parameters={'param': 'value'},
    output_filename="my_validation_results.json"
)
```

**Automatically includes:**
- Results JSON
- Related reports (if any)
- Related plots (if any)

### Manual Upload

If you need to upload reports/plots separately:

```python
from notion_file_uploader import find_related_files
from notion_logger import ValidationRunLogger

logger = ValidationRunLogger()
files = find_related_files("My Validation", Path(".out"))

# Reports are automatically included in create_run_entry()
# Plots are automatically included in create_run_entry()
```

---

## Benefits

1. **Complete Audit Trail:**
   - Results + Analysis + Visualizations all in one place
   - No manual file management
   - Everything linked to validation run

2. **Automatic Discovery:**
   - No need to manually specify files
   - Smart matching by validation type
   - Finds all related content

3. **Consistent Structure:**
   - All runs follow same format
   - Reports formatted consistently
   - Plots organized by validation type

4. **Version Control:**
   - Reports stored in Notion (versioned)
   - Plots referenced (can be updated)
   - Full provenance tracking

---

## Files

- **`notion_file_uploader.py`** - File discovery and upload utilities
- **`notion_logger.py`** - Enhanced to include reports/plots
- **`validation_framework.py`** - Automatic integration

---

## Automatic Syncing

**Three options for automatic Notion syncing:**

1. **GitHub Actions** (recommended) - Auto-syncs on push
   - File: `.github/workflows/notion-sync.yml`
   - Setup: Add `NOTION_API_TOKEN` and `NOTION_VALIDATION_DB_ID` to GitHub Secrets

2. **Post-commit hook** - Syncs immediately after commit
   - File: `.git/hooks/post-commit` (already created)
   - Just run: `chmod +x .git/hooks/post-commit`

3. **File watcher** - Real-time sync during development
   - Run: `python3 notion_sync_watcher.py &`

**Recommendation:** Use GitHub Actions for automatic syncing on push.

---

**This is now the standard practice for all validation runs.**

