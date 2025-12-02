# Cloud Upload Setup for Notion Plots

**Purpose:** Upload plots to cloud storage so they can be embedded directly in Notion pages.

---

## Supported Providers

### 1. Google Cloud Storage (Recommended)

**Advantages:**
- ✅ Automatic public URLs
- ✅ Easy programmatic access
- ✅ Free tier available
- ✅ Reliable and fast

**Setup:**

1. **Create GCS bucket:**
   ```bash
   gsutil mb gs://your-bucket-name
   ```

2. **Set up authentication:**
   
   **Option A: Service Account (Recommended)**
   ```bash
   # Create service account
   gcloud iam service-accounts create notion-uploader \
       --display-name="Notion Plot Uploader"
   
   # Grant storage admin role
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
       --member="serviceAccount:notion-uploader@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
       --role="roles/storage.admin"
   
   # Create and download key
   gcloud iam service-accounts keys create gcs-credentials.json \
       --iam-account=notion-uploader@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

   **Option B: Application Default Credentials**
   ```bash
   gcloud auth application-default login
   ```

3. **Set environment variables:**
   ```bash
   # Add to .env file
   GCS_BUCKET_NAME=your-bucket-name
   GCS_CREDENTIALS_PATH=.env/gcs-credentials.json  # Optional if using ADC
   ```

4. **Make bucket public (for public URLs):**
   ```bash
   gsutil iam ch allUsers:objectViewer gs://your-bucket-name
   ```

### 2. Apple iCloud (Alternative)

**Note:** iCloud upload requires additional setup and may not provide direct public URLs easily.

**Setup:**

1. **Install pyicloud:**
   ```bash
   pip install pyicloud
   ```

2. **Set environment variables:**
   ```bash
   # Add to .env file
   ICLOUD_APPLE_ID=your-apple-id@example.com
   ICLOUD_PASSWORD=your-app-specific-password
   ICLOUD_DRIVE_FOLDER=Notion Uploads
   ```

**Note:** Apple iCloud may require two-factor authentication and app-specific passwords.

---

## Usage

### Automatic (Recommended)

Plots are automatically uploaded when validation runs:

```python
from zetadiffusion.validation_framework import run_validation

# Plots automatically uploaded to cloud and embedded in Notion
result = run_validation(
    validation_type="My Validation",
    validation_func=my_validation
)
```

### Manual Upload

```python
from cloud_uploader import CloudUploader
from pathlib import Path

uploader = CloudUploader(provider="gcs")
plot_url = uploader.upload_plot(Path(".out/plots/my_plot.png"))

if plot_url:
    print(f"Plot uploaded: {plot_url}")
```

---

## Configuration

### Environment Variables

**Google Cloud Storage:**
- `GCS_BUCKET_NAME` - Your GCS bucket name (required)
- `GCS_CREDENTIALS_PATH` - Path to service account JSON (optional if using ADC)

**Apple iCloud:**
- `ICLOUD_APPLE_ID` - Your Apple ID email
- `ICLOUD_PASSWORD` - App-specific password
- `ICLOUD_DRIVE_FOLDER` - Folder name in iCloud Drive (default: "Notion Uploads")

### Auto-Detection

The system automatically detects which provider to use:
1. Checks for `GCS_BUCKET_NAME` → Uses Google Cloud Storage
2. Checks for `ICLOUD_APPLE_ID` → Uses Apple iCloud
3. Falls back to file reference if neither configured

---

## File Organization

Uploaded files are organized in cloud storage:

**Google Cloud Storage:**
```
gs://your-bucket/
  notion-uploads/
    20251201_143022/
      plot1_feg_coherence_vs_chaos.png
      plot2_feg_period_vs_chaos.png
    ...
```

**URL Format:**
```
https://storage.googleapis.com/your-bucket/notion-uploads/20251201_143022/plot1.png
```

---

## Cost Considerations

### Google Cloud Storage

**Free Tier:**
- 5 GB storage
- 5,000 Class A operations/month
- 50,000 Class B operations/month

**Pricing (after free tier):**
- Storage: $0.020/GB/month
- Class A operations: $0.05/10,000
- Class B operations: $0.004/10,000

**For Notion plots:**
- Typical plot: ~200 KB
- 100 plots = ~20 MB
- Well within free tier

### Apple iCloud

- Free tier: 5 GB
- Paid plans: $0.99/month for 50 GB

---

## Security

### Google Cloud Storage

**Public URLs (Recommended for Notion):**
- Files are publicly accessible
- Use for non-sensitive plots/diagrams
- URLs are long and hard to guess

**Private URLs (Alternative):**
- Files are private
- Signed URLs valid for 1 year
- More secure but URLs expire

### Best Practices

1. **Use separate bucket** for Notion uploads
2. **Set lifecycle policy** to delete old files:
   ```bash
   gsutil lifecycle set lifecycle.json gs://your-bucket
   ```
3. **Monitor usage** via GCS console
4. **Use service account** with minimal permissions

---

## Troubleshooting

### GCS Upload Fails

**Error: "Could not initialize GCS client"**
- Check `GCS_CREDENTIALS_PATH` points to valid JSON
- Or run `gcloud auth application-default login`

**Error: "Bucket not found"**
- Verify `GCS_BUCKET_NAME` is correct
- Check bucket exists: `gsutil ls gs://your-bucket`

**Error: "Permission denied"**
- Service account needs `storage.objects.create` permission
- Or use `roles/storage.admin` role

### iCloud Upload Fails

**Error: "Authentication failed"**
- Use app-specific password (not regular password)
- Enable two-factor authentication
- Check Apple ID credentials

---

## Files

- **`cloud_uploader.py`** - Cloud upload implementation
- **`notion_file_uploader.py`** - Enhanced to use cloud URLs
- **`CLOUD_UPLOAD_SETUP.md`** - This file

---

**Once configured, all plots will automatically upload to cloud and embed in Notion pages!**




