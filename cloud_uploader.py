"""
cloud_uploader.py

Uploads files (plots, images) to cloud storage for Notion embedding.
Supports Google Cloud Storage and Apple iCloud (via iCloud Drive API).

Author: Joel
"""

import os
import base64
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

# Google Cloud Storage
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# Apple iCloud (via pyicloud)
try:
    from pyicloud import PyiCloudService
    ICLOUD_AVAILABLE = True
except ImportError:
    ICLOUD_AVAILABLE = False

class CloudUploader:
    """
    Uploads files to cloud storage and returns public URLs for Notion embedding.
    """
    
    def __init__(self, provider: str = "gcs"):
        """
        Initialize cloud uploader.
        
        Args:
            provider: "gcs" (Google Cloud Storage) or "icloud" (Apple iCloud)
        """
        self.provider = provider.lower()
        self.gcs_client = None
        self.gcs_bucket = None
        self.icloud_service = None
        
        if self.provider == "gcs":
            self._init_gcs()
        elif self.provider == "icloud":
            self._init_icloud()
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gcs' or 'icloud'")
    
    def _init_gcs(self):
        """Initialize Google Cloud Storage client."""
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
        
        # Get credentials from environment
        credentials_path = os.getenv("GCS_CREDENTIALS_PATH")
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable not set")
        
        # Initialize client
        if credentials_path and Path(credentials_path).exists():
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.gcs_client = storage.Client(credentials=credentials)
        else:
            # Try default credentials (e.g., from gcloud auth)
            try:
                self.gcs_client = storage.Client()
            except Exception as e:
                raise ValueError(f"Could not initialize GCS client: {e}. Set GCS_CREDENTIALS_PATH or use gcloud auth.")
        
        self.gcs_bucket = self.gcs_client.bucket(bucket_name)
    
    def _init_icloud(self):
        """Initialize Apple iCloud service."""
        if not ICLOUD_AVAILABLE:
            raise ImportError("pyicloud not installed. Install with: pip install pyicloud")
        
        apple_id = os.getenv("ICLOUD_APPLE_ID")
        password = os.getenv("ICLOUD_PASSWORD")
        
        if not apple_id or not password:
            raise ValueError("ICLOUD_APPLE_ID and ICLOUD_PASSWORD environment variables not set")
        
        self.icloud_service = PyiCloudService(apple_id, password)
    
    def upload_file(self, file_path: Path, public: bool = True) -> Optional[str]:
        """
        Upload file to cloud storage and return public URL.
        
        Args:
            file_path: Path to file to upload
            public: Whether to make file publicly accessible (GCS only)
        
        Returns:
            Public URL to uploaded file, or None if upload failed
        """
        if not file_path.exists():
            return None
        
        if self.provider == "gcs":
            return self._upload_gcs(file_path, public)
        elif self.provider == "icloud":
            return self._upload_icloud(file_path)
        else:
            return None
    
    def _upload_gcs(self, file_path: Path, public: bool = True) -> Optional[str]:
        """Upload to Google Cloud Storage."""
        try:
            # Generate unique blob name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"notion-uploads/{timestamp}/{file_path.name}"
            
            # Upload file
            blob = self.gcs_bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
            
            # Make public if requested
            if public:
                blob.make_public()
                return blob.public_url
            else:
                # Generate signed URL (valid for 1 year)
                return blob.generate_signed_url(expiration=31536000)  # 1 year
                
        except Exception as e:
            print(f"Error uploading to GCS: {e}")
            return None
    
    def _upload_icloud(self, file_path: Path) -> Optional[str]:
        """
        Upload to Apple iCloud Drive.
        
        Note: iCloud doesn't provide direct public URLs easily.
        This is a placeholder - would need iCloud Drive API or web interface.
        """
        try:
            # iCloud Drive upload (requires specific folder structure)
            drive_folder = os.getenv("ICLOUD_DRIVE_FOLDER", "Notion Uploads")
            
            # Upload to iCloud Drive
            # Note: This is simplified - actual implementation would need
            # to handle iCloud Drive API properly
            print(f"Warning: iCloud upload not fully implemented. File: {file_path.name}")
            print("Consider using Google Cloud Storage for automatic public URLs.")
            
            # For now, return None - would need proper iCloud Drive API integration
            return None
            
        except Exception as e:
            print(f"Error uploading to iCloud: {e}")
            return None
    
    def upload_plot(self, plot_path: Path) -> Optional[str]:
        """
        Convenience method for uploading plots.
        
        Args:
            plot_path: Path to plot image file
        
        Returns:
            Public URL to uploaded plot, or None if upload failed
        """
        return self.upload_file(plot_path, public=True)

def create_image_block_from_url(image_url: str, caption: str = None) -> dict:
    """
    Create Notion image block from public URL.
    
    Args:
        image_url: Public URL to image
        caption: Optional caption for image
    
    Returns:
        Notion image block
    """
    block = {
        "object": "block",
        "type": "image",
        "image": {
            "type": "external",
            "external": {
                "url": image_url
            }
        }
    }
    
    if caption:
        block["image"]["caption"] = [{"text": {"content": caption}}]
    
    return block

def upload_plot_to_cloud(plot_path: Path, provider: str = None) -> Optional[str]:
    """
    Convenience function to upload plot to cloud and return URL.
    
    Args:
        plot_path: Path to plot file
        provider: Cloud provider ("gcs" or "icloud"), or None to auto-detect from env
    
    Returns:
        Public URL to uploaded plot, or None if upload failed
    """
    # Auto-detect provider from environment
    if provider is None:
        if os.getenv("GCS_BUCKET_NAME"):
            provider = "gcs"
        elif os.getenv("ICLOUD_APPLE_ID"):
            provider = "icloud"
        else:
            print("Warning: No cloud provider configured. Set GCS_BUCKET_NAME or ICLOUD_APPLE_ID")
            return None
    
    try:
        uploader = CloudUploader(provider=provider)
        url = uploader.upload_plot(plot_path)
        return url
    except Exception as e:
        print(f"Warning: Could not upload to cloud: {e}")
        return None

