"""
notion_logger.py

Automates logging validation runs to Notion database with full provenance tracking.

Features:
- Logs validation runs with Cursor session URLs
- Captures execution results and timing
- Uploads visualization files
- Creates complete audit trail from development to results

Author: Joel
"""

import os
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import subprocess

# Import file upload utilities
try:
    from notion_file_uploader import (
        read_markdown_file, markdown_to_notion_blocks,
        create_image_block_from_file, find_related_files
    )
    FILE_UPLOAD_AVAILABLE = True
except ImportError:
    FILE_UPLOAD_AVAILABLE = False
    print("Warning: notion_file_uploader not available - reports/plots won't be uploaded")

try:
    from notion_client import Client
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    print("Warning: notion-client not installed. Install with: pip install notion-client")

# Configuration
NOTION_TOKEN = os.getenv("NOTION_API_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_VALIDATION_DB_ID")  # Your validation runs database ID

class ValidationRunLogger:
    """
    Logs validation runs to Notion with full provenance.
    """
    
    def __init__(self, notion_token: Optional[str] = None, database_id: Optional[str] = None):
        self.notion_token = notion_token or NOTION_TOKEN
        self.database_id = database_id or NOTION_DATABASE_ID
        
        if NOTION_AVAILABLE and self.notion_token:
            self.client = Client(auth=self.notion_token)
        else:
            self.client = None
            if not NOTION_AVAILABLE:
                print("Warning: notion-client package not available")
            if not self.notion_token:
                print("Warning: NOTION_API_TOKEN environment variable not set")
    
    def get_cursor_session_url(self) -> Optional[str]:
        """
        Attempts to extract Cursor session URL from environment or git.
        
        Cursor may expose session info via:
        - Environment variables
        - Git commit messages
        - Session metadata files
        """
        # Check environment variables
        cursor_url = os.getenv("CURSOR_SESSION_URL")
        if cursor_url:
            return cursor_url
        
        # Check for Cursor session file (if Cursor creates one)
        cursor_session_file = Path(".cursor/session_url.txt")
        if cursor_session_file.exists():
            return cursor_session_file.read_text().strip()
        
        # Try to extract from git commit (if session URL was added to commit message)
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                commit_msg = result.stdout
                # Look for Cursor session URL pattern
                import re
                url_pattern = r'https?://[^\s]+cursor[^\s]*'
                match = re.search(url_pattern, commit_msg, re.IGNORECASE)
                if match:
                    return match.group(0)
        except:
            pass
        
        return None
    
    def format_results_summary(self, results: Dict) -> str:
        """Format validation results as readable summary."""
        summary_parts = []
        
        if "delta_estimate" in results:
            summary_parts.append(f"δ estimate: {results['delta_estimate']:.6f}")
            if "delta_error" in results:
                summary_parts.append(f"Error: {results['delta_error']:.6f}")
        
        if "entropy_analysis" in results:
            ent = results["entropy_analysis"]
            summary_parts.append(f"H∞: {ent.get('H_inf', 'N/A'):.3f}")
            summary_parts.append(f"R²: {ent.get('r_squared', 'N/A'):.3f}")
        
        if "transitions" in results:
            summary_parts.append(f"Transitions: {len(results['transitions'])}")
        
        if "fixed_points" in results:
            summary_parts.append(f"Fixed points: {len(results['fixed_points'])}")
        
        if "eigenvalues" in results:
            summary_parts.append(f"Eigenvalues: {len(results['eigenvalues'])}")
        
        return " | ".join(summary_parts) if summary_parts else "No summary available"
    
    def create_run_entry(
        self,
        validation_type: str,
        parameters: Dict,
        results: Dict,
        execution_time: float,
        output_files: List[str] = None,
        cursor_session_url: Optional[str] = None,
        git_commit: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a validation run entry in Notion database.
        
        Args:
            validation_type: Type of validation (e.g., "Temperature Cascade", "Operator Analysis")
            parameters: Validation parameters
            results: Validation results dictionary
            execution_time: Execution time in seconds
            output_files: List of output file paths (visualizations, JSON, etc.)
            cursor_session_url: Cursor session URL (auto-detected if None)
            git_commit: Git commit hash (auto-detected if None)
        
        Returns:
            Notion page ID if successful, None otherwise
        """
        if not self.client or not self.database_id:
            print("Notion client not configured. Skipping log entry.")
            return None
        
        # Auto-detect Cursor session URL if not provided
        if cursor_session_url is None:
            cursor_session_url = self.get_cursor_session_url()
        
        # Auto-detect git commit if not provided
        if git_commit is None:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    git_commit = result.stdout.strip()
            except:
                pass
        
        # Format timestamp
        timestamp = datetime.now().isoformat()
        
        # Get database to check property names and data source
        data_source_id = None
        try:
            db = self.client.databases.retrieve(database_id=self.database_id)
            db_props = db.get('properties', {})
            # Get data source ID if available
            data_sources = db.get('data_sources', [])
            data_source_id = data_sources[0]['id'] if data_sources else None
            
            # Map our property names to actual database property names
            # Try exact match first, then case-insensitive
            def get_prop_name(desired_name):
                if desired_name in db_props:
                    return desired_name
                # Case-insensitive search
                for prop_name in db_props.keys():
                    if prop_name.lower() == desired_name.lower():
                        return prop_name
                return None
            
            # Map all possible property names
            run_id_prop = get_prop_name("Run ID") or get_prop_name("run_id") or get_prop_name("Run-ID")
            validation_type_prop = get_prop_name("Validation Type") or get_prop_name("validation_type")
            timestamp_prop = get_prop_name("Timestamp") or get_prop_name("timestamp") or get_prop_name("Date")
            execution_time_prop = get_prop_name("Execution Time (s)") or get_prop_name("execution_time") or get_prop_name("Execution Time")
            status_prop = get_prop_name("Status") or get_prop_name("status")
            cursor_session_prop = get_prop_name("Cursor Session") or get_prop_name("cursor_session") or get_prop_name("Cursor URL")
            git_commit_prop = get_prop_name("Git Commit") or get_prop_name("git_commit") or get_prop_name("Commit")
            results_summary_prop = get_prop_name("Results Summary") or get_prop_name("results_summary") or get_prop_name("Summary")
        except:
            # Fallback to default names if we can't fetch database
            db_props = {}
            run_id_prop = "Run ID"
            validation_type_prop = "Validation Type"
            timestamp_prop = "Timestamp"
            execution_time_prop = "Execution Time (s)"
            status_prop = "Status"
            cursor_session_prop = "Cursor Session"
            git_commit_prop = "Git Commit"
            results_summary_prop = "Results Summary"
        
        # Generate Run ID (short UUID)
        run_id = str(uuid.uuid4())[:8]
        
        # Determine status
        status = "Completed" if results.get("success", True) else "Failed"
        
        # Create properties for Notion page
        # Database may not have properties yet, so start empty
        properties = {}
        
        # Try to add title if database has title property
        # If no title property, use Run ID or validation type as title
        title_set = False
        if db_props:
            for prop_name, prop_data in db_props.items():
                if prop_data.get('type') == 'title':
                    # Use Run ID in title if available, otherwise validation type
                    title_content = f"{run_id} - {validation_type}" if run_id_prop else f"{validation_type} - {timestamp[:10]}"
                    properties[prop_name] = {
                        "title": [{"text": {"content": title_content}}]
                    }
                    title_set = True
                    break
        
        # Add Run ID if property exists
        if db_props and run_id_prop in db_props:
            prop_type = db_props[run_id_prop].get('type')
            if prop_type == 'title':
                # If Run ID is the title, use it
                properties[run_id_prop] = {
                    "title": [{"text": {"content": run_id}}]
                }
            elif prop_type == 'rich_text':
                properties[run_id_prop] = {
                    "rich_text": [{"text": {"content": run_id}}]
                }
            elif prop_type == 'number':
                # Try to parse as number if it's numeric
                try:
                    properties[run_id_prop] = {"number": int(run_id, 16)}
                except:
                    pass
        
        # Add other properties if they exist in database
        if db_props and validation_type_prop in db_props:
            prop_type = db_props[validation_type_prop].get('type')
            if prop_type == 'select':
                properties[validation_type_prop] = {
                    "select": {"name": validation_type}
                }
            elif prop_type == 'rich_text':
                properties[validation_type_prop] = {
                    "rich_text": [{"text": {"content": validation_type}}]
                }
        
        if db_props and timestamp_prop in db_props:
            prop_type = db_props[timestamp_prop].get('type')
            if prop_type == 'date':
                properties[timestamp_prop] = {
                    "date": {"start": timestamp}
                }
            elif prop_type == 'created_time':
                # Created time is auto-set, skip
                pass
        
        if db_props and execution_time_prop in db_props:
            prop_type = db_props[execution_time_prop].get('type')
            if prop_type == 'number':
                properties[execution_time_prop] = {
                    "number": execution_time
                }
        
        if db_props and status_prop in db_props:
            prop_type = db_props[status_prop].get('type')
            if prop_type == 'select':
                properties[status_prop] = {
                    "select": {"name": status}
                }
            elif prop_type == 'rich_text':
                properties[status_prop] = {
                    "rich_text": [{"text": {"content": status}}]
                }
        
        # Add optional properties only if they exist in database
        if db_props:
            if cursor_session_url and cursor_session_prop in db_props:
                properties[cursor_session_prop] = {
                    "url": cursor_session_url
                }
            
            if git_commit and git_commit_prop in db_props:
                git_url = f"https://github.com/{os.getenv('GITHUB_REPO', 'repo')}/commit/{git_commit}"
                properties[git_commit_prop] = {
                    "url": git_url
                }
            
            results_summary = self.format_results_summary(results)
            if results_summary_prop in db_props:
                properties[results_summary_prop] = {
                    "rich_text": [{"text": {"content": results_summary[:2000]}}]  # Limit to 2000 chars
                }
        
        # Add parameters as JSON in a code block
        parameters_text = json.dumps(parameters, indent=2)
        
        # Create page content
        children = []
        
        # Title as heading (since properties may not exist)
        children.append({
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"text": {"content": f"{validation_type} - {timestamp[:10]}"}}]
            }
        })
        
        # Metadata section with Run ID
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"text": {"content": f"Run ID: {run_id} | "}},
                    {"text": {"content": f"Validation Type: {validation_type} | "}},
                    {"text": {"content": f"Status: {status} | "}},
                    {"text": {"content": f"Execution Time: {execution_time:.3f}s"}}
                ]
            }
        })
        
        # Add Cursor Session and Git Commit if available
        if cursor_session_url or git_commit:
            links_text = []
            if cursor_session_url:
                links_text.append({"text": {"content": f"Cursor Session: {cursor_session_url}", "link": {"url": cursor_session_url}}})
            if git_commit:
                git_url = f"https://github.com/{os.getenv('GITHUB_REPO', 'repo')}/commit/{git_commit}"
                links_text.append({"text": {"content": f" | Git Commit: {git_commit[:8]}", "link": {"url": git_url}}})
            
            if links_text:
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": links_text}
                })
        
        # Parameters section
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"text": {"content": "Parameters"}}]
            }
        })
        
        # Split large content into chunks (Notion limit: 2000 chars per code block)
        max_chunk_size = 1900  # Leave some margin
        if len(parameters_text) > max_chunk_size:
            # Split into multiple code blocks
            for i in range(0, len(parameters_text), max_chunk_size):
                chunk = parameters_text[i:i+max_chunk_size]
                children.append({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "language": "json",
                        "rich_text": [{"text": {"content": chunk}}]
                    }
                })
        else:
            children.append({
                "object": "block",
                "type": "code",
                "code": {
                    "language": "json",
                    "rich_text": [{"text": {"content": parameters_text}}]
                }
            })
        
        # Results section
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"text": {"content": "Results"}}]
            }
        })
        
        results_text = json.dumps(results, indent=2, default=str)
        if len(results_text) > max_chunk_size:
            # Split into multiple code blocks
            for i in range(0, len(results_text), max_chunk_size):
                chunk = results_text[i:i+max_chunk_size]
                children.append({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "language": "json",
                        "rich_text": [{"text": {"content": chunk}}]
                    }
                })
        else:
            children.append({
                "object": "block",
                "type": "code",
                "code": {
                    "language": "json",
                    "rich_text": [{"text": {"content": results_text}}]
                }
            })
        
        # Find and upload related reports and plots
        if FILE_UPLOAD_AVAILABLE:
            output_dir = Path(output_files[0]).parent if output_files else Path(".out")
            related_files = find_related_files(validation_type, output_dir)
            
            # Upload reports (markdown files)
            if related_files['reports']:
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Analysis Reports"}}]
                    }
                })
                
                for report_path in related_files['reports']:
                    try:
                        markdown_content = read_markdown_file(report_path)
                        if markdown_content:
                            report_blocks = markdown_to_notion_blocks(markdown_content)
                            children.extend(report_blocks)
                    except Exception as e:
                        print(f"Warning: Could not upload report {report_path}: {e}")
                        # Fallback: just mention the file
                        children.append({
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{
                                    "text": {"content": f"Report: {report_path.name}"},
                                    "annotations": {"code": True}
                                }]
                            }
                        })
            
            # Upload plots (images)
            if related_files['plots']:
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "Diagnostic Plots"}}]
                    }
                })
                
                for plot_path in related_files['plots']:
                    try:
                        image_block = create_image_block_from_file(plot_path)
                        if image_block:
                            children.append(image_block)
                    except Exception as e:
                        print(f"Warning: Could not upload plot {plot_path}: {e}")
        
        # Output files section
        if output_files:
            children.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "Output Files"}}]
                }
            })
            for file_path in output_files:
                if Path(file_path).exists():
                    file_name = Path(file_path).name
                    children.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{
                                "text": {"content": file_name},
                                "annotations": {"code": True}
                            }]
                        }
                    })
        
        try:
            # Use data_source_id if available (works for databases with data sources)
            if data_source_id:
                parent = {"data_source_id": data_source_id}
            else:
                parent = {"database_id": self.database_id}
            
            try:
                page = self.client.pages.create(
                    parent=parent,
                    properties=properties,
                    children=children
                )
            except Exception as e:
                error_msg = str(e).lower()
                if "multiple data sources" in error_msg:
                    # Multi-source databases aren't supported by the API
                    # Workaround: Create page as child of database's parent page
                    # and add database relation property if available
                    try:
                        db = self.client.databases.retrieve(database_id=self.database_id)
                        parent_page_id = db.get('parent', {}).get('page_id')
                        
                        if parent_page_id:
                            # Create page under parent, then we can manually link to database
                            # But this loses the database structure
                            raise Exception(
                                "Database has multiple data sources which aren't supported by this API version. "
                                "Please use Notion's UI to create entries, or use a database with a single data source. "
                                f"Database: {db.get('title', [{}])[0].get('plain_text', 'Unknown')}"
                            )
                        else:
                            raise
                    except Exception as e2:
                        if "multiple data sources" not in str(e2).lower():
                            raise Exception(f"Could not handle multi-source database. Original: {e}, Secondary: {e2}")
                        raise e2
                else:
                    raise
            
            page_id = page["id"]
            page_url = page.get("url", f"https://notion.so/{page_id.replace('-', '')}")
            
            print(f"✓ Validation run logged to Notion: {page_url}")
            return page_id
            
        except Exception as e:
            print(f"Error creating Notion entry: {e}")
            return None
    
    def log_validation_run(
        self,
        validation_type: str,
        parameters: Dict,
        results: Dict,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Convenience method to log a validation run.
        
        Automatically finds output files in output_dir and includes them.
        """
        # Find output files
        output_files = []
        if output_dir:
            output_path = Path(output_dir)
            if output_path.exists():
                # Look for common output file patterns
                for pattern in ["*.json", "*.png", "*.svg", "*.pdf", "*.txt"]:
                    output_files.extend(output_path.glob(pattern))
        
        # Extract execution time from results if available
        execution_time = results.get("execution_time", 0.0)
        
        return self.create_run_entry(
            validation_type=validation_type,
            parameters=parameters,
            results=results,
            execution_time=execution_time,
            output_files=[str(f) for f in output_files],
            **kwargs
        )


# Decorator for automatic logging
def log_to_notion(validation_type: str, output_dir: Optional[str] = None):
    """
    Decorator to automatically log validation function results to Notion.
    
    Usage:
        @log_to_notion("Temperature Cascade", output_dir=".out")
        def run_validation():
            # ... validation code ...
            return results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Run validation
            results = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Add execution time to results
            if isinstance(results, dict):
                results["execution_time"] = execution_time
                results["success"] = True
            else:
                results = {
                    "result": results,
                    "execution_time": execution_time,
                    "success": True
                }
            
            # Extract parameters from kwargs
            parameters = kwargs.copy()
            
            # Log to Notion
            logger = ValidationRunLogger()
            logger.log_validation_run(
                validation_type=validation_type,
                parameters=parameters,
                results=results,
                output_dir=output_dir
            )
            
            return results
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Example: Log a sample validation run
    logger = ValidationRunLogger()
    
    sample_results = {
        "delta_estimate": 4.669,
        "delta_error": 0.001,
        "transitions": 3,
        "execution_time": 45.2,
        "success": True
    }
    
    sample_parameters = {
        "n_temperatures": 40,
        "temp_min": 0.1,
        "temp_max": 4.0,
        "prompt": "The meaning of life is"
    }
    
    page_id = logger.create_run_entry(
        validation_type="Temperature Cascade",
        parameters=sample_parameters,
        results=sample_results,
        execution_time=45.2,
        output_files=[".out/temperature_cascade_results.json"]
    )
    
    if page_id:
        print(f"Sample entry created with page ID: {page_id}")
    else:
        print("Sample entry creation skipped (Notion not configured)")

