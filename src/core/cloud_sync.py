"""
Cloud Sync Engine
Handles backup and synchronization to AWS S3, Google Cloud Storage, and Azure Blob.
Provides automated backup, integrity verification, and disaster recovery.
"""

import os
import gzip
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class CloudSyncEngine:
    """
    Multi-cloud backup engine for Forex ML Bot.
    Supports: AWS S3, Google Cloud Storage, Azure Blob Storage
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("cloud_backup", {})
        self.enabled = self.config.get("enabled", True)
        self.provider = self.config.get("provider", "aws")
        self._clients = {}
        
        if not self.enabled:
            logger.info("Cloud sync is disabled")
            return
        
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize cloud storage client based on provider"""
        if self.provider == "aws":
            self._init_aws()
        elif self.provider == "gcp":
            self._init_gcp()
        elif self.provider == "azure":
            self._init_azure()
        else:
            logger.warning(f"Unknown cloud provider: {self.provider}")
            self.enabled = False

    def _init_aws(self) -> None:
        """Initialize AWS S3 client"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.bucket = self.config.get("s3_bucket", os.getenv("S3_BUCKET"))
            self.region = self.config.get("aws_region", "us-east-1")
            self.prefix = self.config.get("s3_prefix", "forex-bot/")
            
            if not self.bucket:
                logger.warning("S3 bucket not configured - disabling AWS sync")
                self.enabled = False
                return
            
            self._clients["s3"] = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            self._compress = self.config.get("compression", "gzip") == "gzip"
            self._encrypt = self.config.get("encryption", True)
            
            logger.info(f"AWS S3 client initialized | Bucket: {self.bucket} | Region: {self.region}")
            
        except ImportError:
            logger.error("boto3 not installed - AWS sync disabled")
            self.enabled = False

    def _init_gcp(self) -> None:
        """Initialize Google Cloud Storage client"""
        try:
            from google.cloud import storage
            
            self.bucket = self.config.get("gcs_bucket", os.getenv("GCS_BUCKET"))
            self.prefix = self.config.get("gcs_prefix", "forex-bot/")
            
            if not self.bucket:
                logger.warning("GCS bucket not configured - disabling GCP sync")
                self.enabled = False
                return
            
            self._clients["gcs"] = storage.Client()
            self._compress = self.config.get("compression", "gzip") == "gzip"
            
            logger.info(f"GCS client initialized | Bucket: {self.bucket}")
            
        except ImportError:
            logger.error("google-cloud-storage not installed - GCP sync disabled")
            self.enabled = False

    def _init_azure(self) -> None:
        """Initialize Azure Blob Storage client"""
        try:
            from azure.storage.blob import BlobServiceClient
            
            self.account = self.config.get("azure_account", os.getenv("AZURE_STORAGE_ACCOUNT"))
            self.container = self.config.get("azure_container", "forex-bot")
            self.prefix = self.config.get("azure_prefix", "")
            
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                logger.warning("Azure connection string not configured - disabling Azure sync")
                self.enabled = False
                return
            
            self._clients["azure"] = BlobServiceClient.from_connection_string(connection_string)
            self._compress = self.config.get("compression", "gzip") == "gzip"
            
            logger.info(f"Azure Blob client initialized | Account: {self.account}")
            
        except ImportError:
            logger.error("azure-storage-blob not installed - Azure sync disabled")
            self.enabled = False

    def backup_file(
        self,
        local_path: str,
        remote_subdir: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Backup a single file to cloud storage.
        
        Args:
            local_path: Path to local file
            remote_subdir: Subdirectory in cloud storage
            metadata: Additional metadata to store with file
        
        Returns:
            Remote path of uploaded file
        """
        if not self.enabled:
            logger.info(f"Cloud sync disabled - skipping backup: {local_path}")
            return ""
        
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"File not found: {local}")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        remote_name = f"{local.stem}_{timestamp}{local.suffix}"
        if self._compress:
            remote_name += ".gz"
        
        remote_path = f"{self.prefix}{remote_subdir}/{remote_name}".replace("//", "/")
        
        # Compress if enabled
        if self._compress:
            upload_path = self._compress_file(local)
        else:
            upload_path = str(local)
        
        # Upload
        try:
            if self.provider == "aws":
                self._upload_s3(upload_path, remote_path, metadata)
            elif self.provider == "gcp":
                self._upload_gcs(upload_path, remote_path, metadata)
            elif self.provider == "azure":
                self._upload_azure(upload_path, remote_path, metadata)
            
            logger.info(f"Backed up: {local.name} -> {remote_path}")
            
            # Cleanup temp compressed file
            if self._compress and upload_path != str(local):
                os.remove(upload_path)
            
            return remote_path
            
        except Exception as e:
            logger.error(f"Backup failed for {local.name}: {e}")
            raise

    def _compress_file(self, file_path: Path) -> str:
        """Compress a file using gzip"""
        temp_path = tempfile.mktemp(suffix=".gz")
        with open(file_path, "rb") as f_in:
            with gzip.open(temp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return temp_path

    def _upload_s3(self, local_path: str, remote_path: str, metadata: Optional[Dict] = None) -> None:
        """Upload to AWS S3"""
        extra_args = {}
        if self._encrypt:
            extra_args["ServerSideEncryption"] = "AES256"
        if metadata:
            extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}
        
        self._clients["s3"].upload_file(local_path, self.bucket, remote_path, ExtraArgs=extra_args)

    def _upload_gcs(self, local_path: str, remote_path: str, metadata: Optional[Dict] = None) -> None:
        """Upload to Google Cloud Storage"""
        bucket = self._clients["gcs"].bucket(self.bucket)
        blob = bucket.blob(remote_path)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_filename(local_path)

    def _upload_azure(self, local_path: str, remote_path: str, metadata: Optional[Dict] = None) -> None:
        """Upload to Azure Blob Storage"""
        container = self._clients["azure"].get_container_client(self.container)
        blob = container.get_blob_client(remote_path)
        with open(local_path, "rb") as data:
            blob.upload_blob(data, overwrite=True, metadata=metadata)

    def backup_directory(
        self,
        local_dir: str,
        remote_subdir: str = "",
        pattern: str = "*",
        recursive: bool = True
    ) -> List[str]:
        """Backup all files in a directory"""
        uploaded = []
        local_path = Path(local_dir)
        
        if not local_path.exists():
            logger.warning(f"Directory not found: {local_path}")
            return uploaded
        
        files = local_path.rglob(pattern) if recursive else local_path.glob(pattern)
        
        for file in files:
            if file.is_file():
                try:
                    remote = self.backup_file(str(file), remote_subdir)
                    uploaded.append(remote)
                except Exception as e:
                    logger.error(f"Failed to backup {file}: {e}")
        
        logger.info(f"Directory backup complete: {len(uploaded)} files")
        return uploaded

    def download_file(self, remote_path: str, local_path: str) -> str:
        """Download file from cloud storage"""
        if not self.enabled:
            raise RuntimeError("Cloud sync is disabled")
        
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.provider == "aws":
                self._clients["s3"].download_file(self.bucket, remote_path, str(local))
            elif self.provider == "gcp":
                bucket = self._clients["gcs"].bucket(self.bucket)
                bucket.blob(remote_path).download_to_filename(str(local))
            elif self.provider == "azure":
                container = self._clients["azure"].get_container_client(self.container)
                blob = container.get_blob_client(remote_path)
                with open(local, "wb") as f:
                    f.write(blob.download_blob().readall())
            
            logger.info(f"Downloaded: {remote_path} -> {local}")
            return str(local)
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def list_backups(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List available backups in cloud storage"""
        if not self.enabled:
            return []
        
        search_prefix = f"{self.prefix}{prefix}".rstrip("/") + "/"
        backups = []
        
        try:
            if self.provider == "aws":
                response = self._clients["s3"].list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=search_prefix,
                    MaxKeys=max_keys
                )
                for obj in response.get("Contents", []):
                    backups.append({
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "modified": obj["LastModified"].isoformat(),
                        "etag": obj["ETag"]
                    })
            elif self.provider == "gcp":
                bucket = self._clients["gcs"].bucket(self.bucket)
                for blob in bucket.list_blobs(prefix=search_prefix, max_results=max_keys):
                    backups.append({
                        "key": blob.name,
                        "size": blob.size,
                        "modified": blob.updated.isoformat() if blob.updated else None,
                        "md5": blob.md5_hash
                    })
            elif self.provider == "azure":
                container = self._clients["azure"].get_container_client(self.container)
                for blob in container.list_blobs(name_starts_with=search_prefix):
                    backups.append({
                        "key": blob.name,
                        "size": blob.size,
                        "modified": blob.last_modified.isoformat() if blob.last_modified else None,
                        "etag": blob.etag
                    })
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups

    def verify_backup(self, local_path: str, remote_path: str) -> bool:
        """Verify backup integrity by comparing checksums"""
        try:
            # Calculate local MD5
            local_hash = hashlib.md5()
            with open(local_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    local_hash.update(chunk)
            local_md5 = local_hash.hexdigest()
            
            # Get remote MD5
            if self.provider == "aws":
                response = self._clients["s3"].head_object(Bucket=self.bucket, Key=remote_path)
                remote_md5 = response["ETag"].strip('"')
            elif self.provider == "gcp":
                bucket = self._clients["gcs"].bucket(self.bucket)
                blob = bucket.blob(remote_path)
                blob.reload()
                remote_md5 = blob.md5_hash
            elif self.provider == "azure":
                container = self._clients["azure"].get_container_client(self.container)
                blob = container.get_blob_client(remote_path)
                props = blob.get_blob_properties()
                remote_md5 = hashlib.md5(props.content_settings.content_md5).hexdigest()
            else:
                return False
            
            verified = local_md5 == remote_md5
            logger.info(f"Backup verification: {verified} | {remote_path}")
            return verified
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def cleanup_old_backups(self, days_to_keep: int = 30) -> int:
        """Remove backups older than specified days"""
        if not self.enabled:
            return 0
        
        cutoff = datetime.utcnow() - timedelta(days=days_to_keep)
        deleted = 0
        
        try:
            backups = self.list_backups()
            for backup in backups:
                modified = datetime.fromisoformat(backup["modified"].replace("Z", "+00:00"))
                if modified < cutoff:
                    if self.provider == "aws":
                        self._clients["s3"].delete_object(Bucket=self.bucket, Key=backup["key"])
                    elif self.provider == "gcp":
                        bucket = self._clients["gcs"].bucket(self.bucket)
                        bucket.blob(backup["key"]).delete()
                    elif self.provider == "azure":
                        container = self._clients["azure"].get_container_client(self.container)
                        container.delete_blob(backup["key"])
                    deleted += 1
            
            logger.info(f"Cleanup complete: {deleted} old backups removed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        
        return deleted

    def __repr__(self) -> str:
        return f"CloudSyncEngine(provider={self.provider}, enabled={self.enabled})"
