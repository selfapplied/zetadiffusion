#!.venv/bin/python
"""
setup_notion.py

Interactive setup script for Notion API integration.
Helps configure API token and database ID.
"""

import os
import sys
from pathlib import Path

def get_notion_token():
    """Guide user through getting Notion API token."""
    print("=" * 70)
    print("NOTION API TOKEN SETUP")
    print("=" * 70)
    print("\nTo get your Notion API token:")
    print("1. Go to https://www.notion.so/my-integrations")
    print("2. Click '+ New integration'")
    print("3. Give it a name (e.g., 'Validation Logger')")
    print("4. Select your workspace")
    print("5. Click 'Submit'")
    print("6. Copy the 'Internal Integration Token' (starts with 'secret_')")
    print()
    
    token = input("Paste your Notion API token here (or press Enter to skip): ").strip()
    
    if not token:
        print("⚠ Token not provided. You can set it later with:")
        print("  export NOTION_API_TOKEN='your_token'")
        return None
    
    if not token.startswith('secret_'):
        print("⚠ Warning: Notion tokens usually start with 'secret_'")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return None
    
    return token

def get_database_id():
    """Guide user through getting database ID."""
    print("\n" + "=" * 70)
    print("NOTION DATABASE ID SETUP")
    print("=" * 70)
    print("\nTo get your database ID:")
    print("1. Open your validation runs database in Notion")
    print("2. Click the '...' menu (top right)")
    print("3. Click 'Copy link'")
    print("4. The URL will look like:")
    print("   https://www.notion.so/workspace/Database-Name-abc123def456...")
    print("5. The database ID is the long string at the end (32 characters)")
    print("   It may have dashes: abc123def-4567-8901-2345-6789abcdef01")
    print("   Or be without: abc123def4567890123456789abcdef01")
    print()
    
    db_id = input("Paste your database ID or full URL here: ").strip()
    
    if not db_id:
        print("⚠ Database ID not provided. You can set it later with:")
        print("  export NOTION_VALIDATION_DB_ID='your_database_id'")
        return None
    
    # Extract ID from URL if full URL provided
    if 'notion.so' in db_id:
        # Extract the ID part
        parts = db_id.split('/')
        for part in reversed(parts):
            if len(part) >= 32:
                db_id = part
                break
        # Remove query parameters
        db_id = db_id.split('?')[0]
        # Remove dashes if present
        db_id = db_id.replace('-', '')
        print(f"✓ Extracted database ID: {db_id[:8]}...{db_id[-8:]}")
    
    # Validate format (should be 32 hex characters)
    if len(db_id) != 32:
        print(f"⚠ Warning: Database ID should be 32 characters, got {len(db_id)}")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return None
    
    return db_id

def test_connection(token, db_id):
    """Test Notion API connection."""
    print("\n" + "=" * 70)
    print("TESTING CONNECTION")
    print("=" * 70)
    
    try:
        from notion_client import Client
        
        client = Client(auth=token)
        
        # Try to retrieve database
        try:
            db = client.databases.retrieve(database_id=db_id)
            print(f"✓ Successfully connected to database: {db.get('title', [{}])[0].get('plain_text', 'Unknown')}")
            return True
        except Exception as e:
            print(f"✗ Error accessing database: {e}")
            print("\nCommon issues:")
            print("1. Database ID is incorrect")
            print("2. Integration doesn't have access to the database")
            print("   → Go to database, click '...' → 'Connections' → Add your integration")
            return False
            
    except ImportError:
        print("✗ notion-client package not installed")
        print("  Install with: pip install notion-client")
        return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

def save_to_env_file(token, db_id):
    """Save credentials to .env file."""
    env_file = Path(".env")
    
    lines = []
    if env_file.exists():
        lines = env_file.read_text().split('\n')
        # Remove existing entries
        lines = [l for l in lines if not l.startswith('NOTION_API_TOKEN') and 
                not l.startswith('NOTION_VALIDATION_DB_ID')]
    
    if token:
        lines.append(f"NOTION_API_TOKEN={token}")
    if db_id:
        lines.append(f"NOTION_VALIDATION_DB_ID={db_id}")
    
    env_file.write_text('\n'.join(lines) + '\n')
    print(f"\n✓ Saved to {env_file}")
    print("  Note: Add .env to .gitignore to keep credentials safe!")

def update_gitignore():
    """Ensure .env is in .gitignore."""
    gitignore = Path(".gitignore")
    
    if not gitignore.exists():
        gitignore.write_text(".env\n")
        print("✓ Created .gitignore with .env")
        return
    
    content = gitignore.read_text()
    if '.env' not in content:
        gitignore.write_text(content + "\n.env\n")
        print("✓ Added .env to .gitignore")

def main():
    print("\n" + "=" * 70)
    print("NOTION INTEGRATION SETUP")
    print("=" * 70)
    print("\nThis script will help you configure Notion API access")
    print("for automatic validation run logging.\n")
    
    # Get credentials
    token = get_notion_token()
    db_id = get_database_id()
    
    if not token or not db_id:
        print("\n⚠ Setup incomplete. You can run this script again later.")
        return
    
    # Test connection
    if test_connection(token, db_id):
        print("\n✓ Connection successful!")
        
        # Save to .env file
        save_to_env_file(token, db_id)
        update_gitignore()
        
        # Also set environment variables for current session
        os.environ['NOTION_API_TOKEN'] = token
        os.environ['NOTION_VALIDATION_DB_ID'] = db_id
        
        print("\n" + "=" * 70)
        print("SETUP COMPLETE!")
        print("=" * 70)
        print("\nCredentials saved to .env file")
        print("Environment variables set for this session")
        print("\nTo use in future sessions, either:")
        print("1. Source the .env file: source .env (or use python-dotenv)")
        print("2. Export manually:")
        print(f"   export NOTION_API_TOKEN='{token[:10]}...'")
        print(f"   export NOTION_VALIDATION_DB_ID='{db_id}'")
        print("\nYou can now run: python3 upload_to_notion.py")
    else:
        print("\n⚠ Setup incomplete. Please check your credentials and try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)

