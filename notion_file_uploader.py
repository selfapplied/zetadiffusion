"""
notion_file_uploader.py

Utilities for uploading files (reports, plots) to Notion pages.
Handles markdown reports and image plots.

Author: Joel
"""

import base64
from pathlib import Path
from typing import List, Optional
import re

def read_markdown_file(file_path: Path) -> str:
    """Read markdown file content."""
    try:
        return file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return ""

def markdown_to_notion_blocks(markdown_content: str) -> List[dict]:
    """
    Convert markdown content to Notion blocks.
    
    Handles:
    - Headers (# ## ###)
    - Paragraphs
    - Code blocks (```)
    - Lists (- *)
    - Bold (**text**)
    - Italic (*text*)
    - Links ([text](url))
    """
    blocks = []
    lines = markdown_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Headers
        if line.startswith('###'):
            content = line[3:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"text": {"content": content}}]
                }
            })
        elif line.startswith('##'):
            content = line[2:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": content}}]
                }
            })
        elif line.startswith('#'):
            content = line[1:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"text": {"content": content}}]
                }
            })
        # Code blocks
        elif line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            code_content = '\n'.join(code_lines)
            blocks.append({
                "object": "block",
                "type": "code",
                "code": {
                    "rich_text": [{"text": {"content": code_content}}],
                    "language": "plain text"
                }
            })
        # Lists
        elif line.startswith('- ') or line.startswith('* '):
            list_items = []
            while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                item_content = lines[i].strip()[2:].strip()
                list_items.append(item_content)
                i += 1
            i -= 1  # Adjust for outer loop increment
            
            for item in list_items:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"text": {"content": item}}]
                    }
                })
        # Regular paragraph
        else:
            # Parse inline formatting
            rich_text = parse_inline_formatting(line)
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": rich_text
                }
            })
        
        i += 1
    
    return blocks

def parse_inline_formatting(text: str) -> List[dict]:
    """
    Parse markdown inline formatting to Notion rich text.
    
    Handles:
    - **bold**
    - *italic*
    - [link](url)
    """
    rich_text = []
    
    # Simple parser - handles bold, italic, links
    # Split by markdown patterns
    parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|\[.*?\]\(.*?\))', text)
    
    for part in parts:
        if not part:
            continue
        
        # Bold
        if part.startswith('**') and part.endswith('**'):
            content = part[2:-2]
            rich_text.append({
                "text": {"content": content},
                "annotations": {"bold": True}
            })
        # Italic
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            content = part[1:-1]
            rich_text.append({
                "text": {"content": content},
                "annotations": {"italic": True}
            })
        # Link
        elif part.startswith('[') and '](' in part:
            match = re.match(r'\[(.*?)\]\((.*?)\)', part)
            if match:
                link_text, link_url = match.groups()
                rich_text.append({
                    "text": {"content": link_text},
                    "annotations": {},
                    "href": link_url
                })
            else:
                rich_text.append({"text": {"content": part}})
        else:
            rich_text.append({"text": {"content": part}})
    
    # If no formatting found, return simple text
    if not rich_text:
        rich_text = [{"text": {"content": text}}]
    
    return rich_text

def image_to_base64(file_path: Path) -> Optional[str]:
    """Convert image file to base64 string."""
    try:
        with open(file_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return base64_data
    except Exception as e:
        print(f"Warning: Could not encode {file_path}: {e}")
        return None

def create_image_block_from_file(file_path: Path, upload_to_cloud: bool = True) -> Optional[dict]:
    """
    Create Notion image block from local file.
    
    Attempts to upload to cloud storage (Google Cloud or Apple iCloud) if configured,
    then creates an image block with the public URL.
    
    Falls back to callout block with file reference if cloud upload fails.
    
    Args:
        file_path: Path to image file
        upload_to_cloud: Whether to attempt cloud upload
    
    Returns:
        Notion image block (with URL) or callout block (with file reference)
    """
    if not file_path.exists():
        return None
    
    file_name = file_path.name
    file_size = file_path.stat().st_size / 1024  # KB
    
    # Try cloud upload if enabled
    if upload_to_cloud:
        try:
            from cloud_uploader import upload_plot_to_cloud, create_image_block_from_url
            
            # Upload to cloud and get URL
            image_url = upload_plot_to_cloud(file_path)
            
            if image_url:
                # Create image block with URL
                return create_image_block_from_url(image_url, caption=file_name)
        except ImportError:
            # cloud_uploader not available
            pass
        except Exception as e:
            print(f"Warning: Cloud upload failed for {file_name}: {e}")
    
    # Fallback: Create callout block with file reference
    # Note: Notion callout rich_text doesn't support annotations, use plain text
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [
                {"text": {"content": f"📊 Plot: {file_name} ({file_size:.1f} KB)"}}
            ],
            "icon": {"emoji": "📊"}
        }
    }

def find_related_files(validation_type: str, output_dir: Path) -> dict:
    """
    Find related report and plot files for a validation.
    
    Note: Reports are documentation markdown files (not auto-generated).
    Plots are auto-generated PNG files from validation runs.
    
    Returns:
        {
            'reports': [Path, ...],  # Documentation markdown files
            'plots': [Path, ...]     # Auto-generated plot images
        }
    """
    reports = []
    plots = []
    
    # Normalize validation type for filename matching
    safe_name = validation_type.lower().replace(" ", "_").replace(".", "").replace("-", "_")
    
    # Map validation types to keywords
    type_keywords = {
        'conjecture': ['conjecture', '9_1', '9.1'],
        'feg': ['feg', 'cascade'],
        'temperature': ['temperature', 'cascade'],
        'operator': ['operator', 'analysis'],
        'clock': ['clock', 'execution']
    }
    
    # Find keywords in validation type
    keywords = []
    for key, vals in type_keywords.items():
        if any(v in safe_name for v in vals):
            keywords.extend(vals)
    
    # Find markdown reports in project root
    report_patterns = [
        "*FINDINGS*.md",
        "*findings*.md",
        "*ANALYSIS*.md",
        "*analysis*.md",
        "*SUMMARY*.md",
        "*summary*.md"
    ]
    
    for pattern in report_patterns:
        reports.extend(Path(".").glob(pattern))
    
    # Also check for specific validation reports
    if 'conjecture' in safe_name:
        reports.extend(Path(".").glob("*CONJECTURE*.md"))
        reports.extend(Path(".").glob("*conjecture*.md"))
    if 'clock' in safe_name or 'execution' in safe_name:
        reports.extend(Path(".").glob("*CLOCK*.md"))
        reports.extend(Path(".").glob("*clock*.md"))
    
    # Find plots in .out/plots directory
    plot_dir = Path(".out/plots")
    if plot_dir.exists():
        # Get all plots - we'll filter by relevance
        all_plots = list(plot_dir.glob("*.png"))
        
        # Filter by validation type keywords
        for plot in all_plots:
            plot_name_lower = plot.name.lower()
            # Match if any keyword appears in plot name
            if any(kw in plot_name_lower for kw in keywords) or len(keywords) == 0:
                plots.append(plot)
    
    # Also check for plots in output dir
    if output_dir.exists():
        plots.extend(output_dir.glob("*.png"))
        plots.extend(output_dir.glob("*.svg"))
    
    # Remove duplicates and non-existent files
    reports = [r for r in set(reports) if r.exists()]
    plots = [p for p in set(plots) if p.exists()]
    
    # Sort for consistency
    reports.sort()
    plots.sort()
    
    return {
        'reports': reports,
        'plots': plots
    }

