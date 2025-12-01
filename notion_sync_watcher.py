#!.venv/bin/python
"""
notion_sync_watcher.py

Watches .out/ directory for new validation results and automatically syncs to Notion.

Run in background:
    python3 notion_sync_watcher.py &

Or use systemd/tmux for persistent watching.

Author: Joel
"""

import time
import json
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os

class NotionSyncHandler(FileSystemEventHandler):
    """Watch for new validation result files and sync to Notion."""
    
    def __init__(self):
        self.synced_files = set()
        self.last_sync = time.time()
        self.sync_cooldown = 5.0  # Wait 5 seconds before syncing (batch multiple files)
    
    def on_created(self, event):
        """Handle new file creation."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only watch for result files
        if file_path.suffix == '.json' and '_results.json' in file_path.name:
            print(f"📊 New result file detected: {file_path.name}")
            self.schedule_sync()
    
    def on_modified(self, event):
        """Handle file modifications."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only watch for result files
        if file_path.suffix == '.json' and '_results.json' in file_path.name:
            # Check if file is complete (not being written)
            try:
                with open(file_path, 'r') as f:
                    json.load(f)  # Try to parse - if it fails, file is still being written
                self.schedule_sync()
            except (json.JSONDecodeError, IOError):
                pass  # File still being written, skip
    
    def schedule_sync(self):
        """Schedule a sync after cooldown period."""
        current_time = time.time()
        time_since_last_sync = current_time - self.last_sync
        
        if time_since_last_sync >= self.sync_cooldown:
            self.sync_to_notion()
        else:
            # Schedule sync after cooldown
            remaining = self.sync_cooldown - time_since_last_sync
            print(f"⏳ Waiting {remaining:.1f}s before syncing (cooldown)...")
    
    def sync_to_notion(self):
        """Sync new results to Notion."""
        print("🔄 Syncing to Notion...")
        
        try:
            # Load environment
            env = os.environ.copy()
            if Path('.env').exists():
                with open('.env', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env[key] = value
            
            # Run upload script
            result = subprocess.run(
                ['python3', 'upload_to_notion.py'],
                capture_output=True,
                text=True,
                env=env,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✓ Synced to Notion successfully")
            else:
                print(f"⚠ Sync failed: {result.stderr[:200]}")
            
            self.last_sync = time.time()
            
        except Exception as e:
            print(f"⚠ Sync error: {e}")

def main():
    """Start watching for new validation results."""
    print("=" * 70)
    print("Notion Sync Watcher")
    print("=" * 70)
    print()
    print("Watching .out/ directory for new validation results...")
    print("Press Ctrl+C to stop")
    print()
    
    # Create observer
    event_handler = NotionSyncHandler()
    observer = Observer()
    
    # Watch .out directory
    out_dir = Path(".out")
    out_dir.mkdir(exist_ok=True)
    
    observer.schedule(event_handler, str(out_dir), recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping watcher...")
        observer.stop()
    
    observer.join()
    print("✓ Watcher stopped")

if __name__ == "__main__":
    main()

