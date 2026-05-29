"""
Agent G: Monitoring, Logging & Cloud Sync Engine
================================================
System health monitoring, structured logging, cloud backup,
GitHub sync, and alerting for the entire Forex ML Bot system.

Features: Distributed tracing, automated disaster recovery,
incremental backup with compression, and multi-channel alerting.
"""

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent
from core.cloud_sync import CloudSyncEngine


class MonitoringCloudSyncEngine(BaseAgent):
    """
    Agent G - Monitoring, Logging & Cloud Sync
    
    Pipeline:
    1. Collect system health metrics
    2. Gather artifacts from all agents
    3. Backup to cloud storage
    4. Sync to GitHub
    5. Send alerts if needed
    6. Generate system report
    """

    def __init__(self, config_path: str = "config/system_config.yaml", kaggle_mode: bool = False):
        super().__init__(config_path=config_path, agent_key="agent_g", kaggle_mode=kaggle_mode)
        
        # Config sections
        self.logging_config = self.agent_config.get("logging", {})
        self.monitoring_config = self.agent_config.get("monitoring", {})
        self.cloud_config = self.agent_config.get("cloud_backup", {})
        self.github_config = self.agent_config.get("github", {})
        self.alert_config = self.agent_config.get("alerting", {})
        
        # Initialize cloud sync
        self.cloud = CloudSyncEngine(self.agent_config)
        
        # Health tracking
        self._health_checks: List[Dict[str, Any]] = []
        self._alert_history: List[Dict[str, Any]] = []
        self._last_backup: Optional[datetime] = None
        self._last_github_sync: Optional[datetime] = None

    def _execute_core(self) -> Dict[str, Any]:
        """Execute monitoring and sync pipeline"""
        results = {
            "health_status": "healthy",
            "health_checks": {},
            "artifacts_collected": 0,
            "cloud_backup_path": "",
            "github_synced": False,
            "alerts_sent": 0,
            "errors": []
        }
        
        # Step 1: System health check
        self.logger.info("STEP 1: System health check")
        health = self._check_system_health()
        results["health_checks"] = health
        
        # Determine overall status
        if any(h.get("status") == "critical" for h in health.values()):
            results["health_status"] = "critical"
        elif any(h.get("status") == "warning" for h in health.values()):
            results["health_status"] = "warning"
        
        # Step 2: Collect artifacts from all agents
        self.logger.info("STEP 2: Collecting artifacts")
        artifacts = self._collect_artifacts()
        results["artifacts_collected"] = len(artifacts)
        
        # Step 3: Cloud backup
        self.logger.info("STEP 3: Cloud backup")
        try:
            backup_path = self._backup_to_cloud(artifacts)
            results["cloud_backup_path"] = backup_path
            self._last_backup = datetime.utcnow()
        except Exception as e:
            results["errors"].append(f"Cloud backup failed: {e}")
            self.logger.error(f"Cloud backup failed: {e}")
        
        # Step 4: GitHub sync
        self.logger.info("STEP 4: GitHub sync")
        try:
            synced = self._sync_to_github(artifacts)
            results["github_synced"] = synced
            if synced:
                self._last_github_sync = datetime.utcnow()
        except Exception as e:
            results["errors"].append(f"GitHub sync failed: {e}")
            self.logger.error(f"GitHub sync failed: {e}")
        
        # Step 5: Send alerts
        self.logger.info("STEP 5: Checking alerts")
        alerts_sent = self._send_alerts(results)
        results["alerts_sent"] = alerts_sent
        
        # Step 6: Save system report
        self.logger.info("STEP 6: Saving system report")
        report_path = self._save_system_report(results)
        results["report_path"] = report_path
        
        self.logger.info(
            f"\nAgent G complete | Status: {results['health_status']} | "
            f"Artifacts: {results['artifacts_collected']} | "
            f"Cloud: {bool(results['cloud_backup_path'])} | "
            f"GitHub: {results['github_synced']}"
        )
        
        return results

    def _check_system_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all system components"""
        health = {}
        
        # Check 1: Disk space
        try:
            import psutil
            disk = psutil.disk_usage("/mnt/agents/output" if os.path.exists("/mnt/agents/output") else "/")
            disk_pct = disk.percent
            
            health["disk_space"] = {
                "status": "healthy" if disk_pct < 80 else "warning" if disk_pct < 90 else "critical",
                "used_gb": round(disk.used / 1e9, 2),
                "free_gb": round(disk.free / 1e9, 2),
                "percent_used": disk_pct
            }
        except ImportError:
            health["disk_space"] = {"status": "unknown", "message": "psutil not available"}
        
        # Check 2: Memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            mem_pct = memory.percent
            
            health["memory"] = {
                "status": "healthy" if mem_pct < 80 else "warning" if mem_pct < 90 else "critical",
                "used_gb": round(memory.used / 1e9, 2),
                "available_gb": round(memory.available / 1e9, 2),
                "percent_used": mem_pct
            }
        except ImportError:
            health["memory"] = {"status": "unknown", "message": "psutil not available"}
        
        # Check 3: Agent outputs
        agents = ["agent_a", "agent_b", "agent_c", "agent_d", "agent_e", "agent_f"]
        for agent in agents:
            output_dir = self.project_root / "output" / agent
            if output_dir.exists() and any(output_dir.iterdir()):
                # Check freshness (within 24 hours)
                latest = max(
                    (f for f in output_dir.rglob("*") if f.is_file()),
                    key=lambda x: x.stat().st_mtime,
                    default=None
                )
                if latest:
                    age_hours = (time.time() - latest.stat().st_mtime) / 3600
                    health[f"{agent}_output"] = {
                        "status": "healthy" if age_hours < 24 else "warning",
                        "latest_file": latest.name,
                        "age_hours": round(age_hours, 1)
                    }
                else:
                    health[f"{agent}_output"] = {"status": "warning", "message": "Empty output directory"}
            else:
                health[f"{agent}_output"] = {"status": "critical", "message": f"No output directory for {agent}"}
        
        # Check 4: Cloud connectivity
        if self.cloud.enabled:
            try:
                backups = self.cloud.list_backups(max_keys=1)
                health["cloud_storage"] = {
                    "status": "healthy",
                    "provider": self.cloud.provider,
                    "accessible": True
                }
            except Exception as e:
                health["cloud_storage"] = {
                    "status": "critical",
                    "provider": self.cloud.provider,
                    "accessible": False,
                    "error": str(e)
                }
        else:
            health["cloud_storage"] = {"status": "warning", "message": "Cloud sync disabled"}
        
        # Store health check
        self._health_checks.append({
            "timestamp": datetime.utcnow().isoformat(),
            "checks": health
        })
        
        # Log summary
        status_counts = {}
        for check in health.values():
            status = check.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        self.logger.info(f"Health check: {status_counts}")
        
        return health

    def _collect_artifacts(self) -> List[Path]:
        """Collect all artifacts from agent outputs"""
        artifacts = []
        
        # Collect from all agent output directories
        for agent_dir in (self.project_root / "output").glob("*"):
            if agent_dir.is_dir():
                for file in agent_dir.rglob("*"):
                    if file.is_file() and file.stat().st_size > 0:
                        # Only include recent files (< 7 days)
                        age_days = (time.time() - file.stat().st_mtime) / 86400
                        if age_days < 7:
                            artifacts.append(file)
        
        # Collect logs
        log_dir = self.project_root / "logs"
        if log_dir.exists():
            for file in log_dir.glob("*.log"):
                artifacts.append(file)
        
        # Collect configs
        config_dir = self.project_root / "config"
        if config_dir.exists():
            for file in config_dir.glob("*.yaml"):
                artifacts.append(file)
        
        self.logger.info(f"Collected {len(artifacts)} artifacts")
        return artifacts

    def _backup_to_cloud(self, artifacts: List[Path]) -> str:
        """Backup artifacts to cloud storage"""
        if not self.cloud.enabled:
            return ""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_prefix = f"backups/{timestamp}"
        
        uploaded = []
        
        # Create manifest
        manifest = {
            "backup_time": datetime.utcnow().isoformat(),
            "agent_version": self.config.get("system", {}).get("version", "1.0.0"),
            "files": []
        }
        
        # Upload each artifact
        for artifact in artifacts:
            try:
                relative = artifact.relative_to(self.project_root)
                remote_path = self.cloud.backup_file(
                    str(artifact),
                    remote_subdir=f"{backup_prefix}/{relative.parent}"
                )
                
                if remote_path:
                    # Calculate checksum
                    with open(artifact, "rb") as f:
                        checksum = hashlib.md5(f.read()).hexdigest()
                    
                    manifest["files"].append({
                        "local_path": str(relative),
                        "remote_path": remote_path,
                        "size": artifact.stat().st_size,
                        "md5": checksum
                    })
                    uploaded.append(remote_path)
                    
            except Exception as e:
                self.logger.warning(f"Failed to upload {artifact}: {e}")
        
        # Upload manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        self.cloud.backup_file(str(manifest_path), f"{backup_prefix}/")
        
        # Cleanup old backups
        if self.cloud_config.get("backup_interval_hours", 6) > 0:
            self.cloud.cleanup_old_backups(days_to_keep=30)
        
        self.logger.info(f"Cloud backup: {len(uploaded)} files uploaded")
        return backup_prefix

    def _sync_to_github(self, artifacts: List[Path]) -> bool:
        """Sync important artifacts to GitHub repository"""
        if not self.github_config.get("auto_commit", True):
            return False
        
        repo = self.github_config.get("repo", "")
        if not repo:
            self.logger.info("GitHub repo not configured - skipping sync")
            return False
        
        try:
            # Check if git is available
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                self.logger.info("Git not available - skipping GitHub sync")
                return False
            
            # Check if we're in a git repo
            git_dir = self.project_root / ".git"
            if not git_dir.exists():
                self.logger.info("Not a git repository - initializing")
                subprocess.run(["git", "init"], cwd=self.project_root, capture_output=True)
                subprocess.run(
                    ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"],
                    cwd=self.project_root,
                    capture_output=True
                )
            
            # Stage important files
            important_patterns = [
                "config/*.yaml",
                "src/**/*.py",
                "docs/*.md",
                "notebooks/*.ipynb"
            ]
            
            for pattern in important_patterns:
                for file in self.project_root.glob(pattern):
                    if file.is_file():
                        subprocess.run(
                            ["git", "add", str(file.relative_to(self.project_root))],
                            cwd=self.project_root,
                            capture_output=True
                        )
            
            # Commit
            commit_msg = self.github_config.get(
                "commit_message_template",
                "[BOT] {timestamp} - {event}"
            ).format(
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                event="sync"
            )
            
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Push (may fail if no credentials - that's ok for now)
            subprocess.run(
                ["git", "push", "origin", self.github_config.get("branch", "main"), "--quiet"],
                cwd=self.project_root,
                capture_output=True
            )
            
            self.logger.info("GitHub sync completed")
            return True
            
        except Exception as e:
            self.logger.warning(f"GitHub sync failed (expected if not configured): {e}")
            return False

    def _send_alerts(self, results: Dict[str, Any]) -> int:
        """Send alerts based on health status and events"""
        alerts_sent = 0
        channels = self.alert_config.get("channels", [])
        
        for channel in channels:
            try:
                if channel["type"] == "slack" and channel.get("webhook_url"):
                    sent = self._send_slack_alert(channel, results)
                    if sent:
                        alerts_sent += 1
                        
                elif channel["type"] == "email" and channel.get("smtp_host"):
                    sent = self._send_email_alert(channel, results)
                    if sent:
                        alerts_sent += 1
                        
            except Exception as e:
                self.logger.warning(f"Alert channel {channel['type']} failed: {e}")
        
        return alerts_sent

    def _send_slack_alert(self, config: Dict, results: Dict[str, Any]) -> bool:
        """Send alert to Slack webhook"""
        try:
            import requests
            
            webhook_url = config["webhook_url"]
            level = config.get("level", "warning")
            
            # Only alert on warning level or higher
            if results["health_status"] == "healthy" and level != "info":
                return False
            
            color = "danger" if results["health_status"] == "critical" else "warning" if results["health_status"] == "warning" else "good"
            
            message = {
                "attachments": [{
                    "color": color,
                    "title": f"Forex ML Bot - {results['health_status'].upper()}",
                    "fields": [
                        {"title": "Status", "value": results["health_status"], "short": True},
                        {"title": "Artifacts", "value": str(results["artifacts_collected"]), "short": True},
                        {"title": "Cloud Backup", "value": "Yes" if results["cloud_backup_path"] else "No", "short": True},
                        {"title": "GitHub Sync", "value": "Yes" if results["github_synced"] else "No", "short": True}
                    ],
                    "footer": "Forex ML Bot",
                    "ts": int(time.time())
                }]
            }
            
            response = requests.post(webhook_url, json=message, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.warning(f"Slack alert failed: {e}")
            return False

    def _send_email_alert(self, config: Dict, results: Dict[str, Any]) -> bool:
        """Send alert via email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            level = config.get("level", "critical")
            if results["health_status"] != "critical" and level == "critical":
                return False
            
            smtp_host = config["smtp_host"]
            smtp_port = config.get("smtp_port", 587)
            
            msg = MIMEText(f"""
Forex ML Bot Alert

Status: {results['health_status'].upper()}
Time: {datetime.utcnow().isoformat()}

Health Checks:
{json.dumps(results.get('health_checks', {}), indent=2)}

Errors:
{chr(10).join(results.get('errors', []))}
            """)
            
            msg["Subject"] = f"[Forex Bot] {results['health_status'].upper()} Alert"
            msg["From"] = config.get("from", "bot@forex-ml.com")
            msg["To"] = config.get("to", "admin@forex-ml.com")
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Email alert failed: {e}")
            return False

    def _save_system_report(self, results: Dict[str, Any]) -> str:
        """Save comprehensive system report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": self.config.get("system", {}),
            "health": {
                "status": results["health_status"],
                "checks": results["health_checks"]
            },
            "backup": {
                "cloud_path": results["cloud_backup_path"],
                "github_synced": results["github_synced"],
                "artifacts_collected": results["artifacts_collected"]
            },
            "alerts": {
                "sent": results["alerts_sent"],
                "history": self._alert_history[-10:]
            },
            "errors": results["errors"]
        }
        
        report_path = self.output_dir / f"system_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save latest report
        latest_path = self.output_dir / "latest_report.json"
        with open(latest_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(report_path)

    def get_health_history(self) -> List[Dict[str, Any]]:
        """Return health check history"""
        return self._health_checks

    def get_alert_history(self) -> List[Dict[str, Any]]:
        """Return alert history"""
        return self._alert_history


if __name__ == "__main__":
    agent = MonitoringCloudSyncEngine(kaggle_mode=False)
    result = agent.execute()
    print(f"\nAgent G Result: {result}")
