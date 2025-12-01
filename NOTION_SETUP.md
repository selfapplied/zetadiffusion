# Notion Integration Setup Guide

## Quick Start

Run the interactive setup script:
```bash
python3 setup_notion.py
```

This will guide you through:
1. Getting your Notion API token
2. Finding your database ID
3. Testing the connection
4. Saving credentials securely

## Manual Setup

### Step 1: Get Notion API Token

1. Go to https://www.notion.so/my-integrations
2. Click **"+ New integration"**
3. Name it (e.g., "Validation Logger")
4. Select your workspace
5. Click **"Submit"**
6. Copy the **"Internal Integration Token"** (starts with `secret_`)

### Step 2: Create/Find Your Database

If you don't have a validation runs database yet:

1. Create a new database in Notion
2. Add these properties:
   - **Validation Type** (Select)
   - **Timestamp** (Date)
   - **Execution Time (s)** (Number)
   - **Status** (Select: Completed, Failed)
   - **Cursor Session** (URL) - *This is the new field you added*
   - **Git Commit** (URL)
   - **Results Summary** (Text)
   - **Parameters** (can be in page content)
   - **Results** (can be in page content)

To get the database ID:

1. Open your database in Notion
2. Click the **"..."** menu (top right)
3. Click **"Copy link"**
4. The URL looks like:
   ```
   https://www.notion.so/workspace/Database-Name-abc123def456...
   ```
5. The database ID is the 32-character string at the end
   - With dashes: `abc123def-4567-8901-2345-6789abcdef01`
   - Without: `abc123def4567890123456789abcdef01`

### Step 3: Grant Database Access

**Important:** Your integration needs access to the database!

1. Open your database in Notion
2. Click **"..."** menu → **"Connections"**
3. Find your integration and click to add it
4. The integration should now appear in the connections list

### Step 4: Set Environment Variables

#### Option A: Using .env file (Recommended)

The setup script creates a `.env` file automatically. To use it:

```bash
# Install python-dotenv (optional, for auto-loading)
pip install python-dotenv

# Or manually source it
export $(cat .env | xargs)
```

#### Option B: Manual export

```bash
export NOTION_API_TOKEN='secret_your_token_here'
export NOTION_VALIDATION_DB_ID='your_32_character_database_id'
```

#### Option C: Add to shell profile

Add to `~/.zshrc` or `~/.bashrc`:
```bash
export NOTION_API_TOKEN='secret_your_token_here'
export NOTION_VALIDATION_DB_ID='your_32_character_database_id'
```

Then reload:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

### Step 5: Test Connection

```bash
python3 setup_notion.py
```

Or test manually:
```bash
python3 -c "
from notion_logger import ValidationRunLogger
logger = ValidationRunLogger()
if logger.client:
    print('✓ Notion client configured')
else:
    print('✗ Notion client not configured')
"
```

## Troubleshooting

### "Database not found" error

- Check that the database ID is correct (32 characters)
- Ensure your integration has access to the database
- Go to database → Connections → Add your integration

### "Invalid token" error

- Verify token starts with `secret_`
- Check that token hasn't been revoked
- Create a new integration if needed

### "Permission denied" error

- Your integration needs to be added to the database
- Go to database → Connections → Add integration

### Environment variables not loading

- Check they're set: `echo $NOTION_API_TOKEN`
- If using .env, ensure you're sourcing it or using python-dotenv
- Try exporting manually in the same terminal session

## Database Schema Recommendations

Your Notion database should have these properties:

| Property Name | Type | Description |
|--------------|------|-------------|
| Validation Type | Select | e.g., "Operator Analysis", "Temperature Cascade" |
| Timestamp | Date | When the run was executed |
| Execution Time (s) | Number | Runtime in seconds |
| Status | Select | "Completed" or "Failed" |
| Cursor Session | URL | Link to Cursor AI session |
| Git Commit | URL | Link to git commit |
| Results Summary | Text | Brief summary of results |

Additional content (Parameters, Results, Output Files) can be stored in the page body as code blocks.

## Next Steps

Once configured, you can:

1. **Upload existing results:**
   ```bash
   python3 upload_to_notion.py
   ```

2. **Run new validations with auto-logging:**
   ```bash
   python3 save_analysis_results.py
   python3 upload_to_notion.py
   ```

3. **Use the decorator in your code:**
   ```python
   from notion_logger import log_to_notion
   
   @log_to_notion("My Validation", output_dir=".out")
   def my_validation():
       # ... your code ...
       return results
   ```

## Security Notes

- ✅ `.env` file is automatically added to `.gitignore`
- ✅ Never commit API tokens to git
- ✅ Tokens can be revoked and recreated in Notion
- ✅ Each integration is workspace-specific

