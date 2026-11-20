"""Persistent SQLite Alert Storage for Explosion and Fireworks Detections.

Provides thread-safe persistence, querying, and filtering of real-time alerts
in reports/alerts_history.db without external database dependencies.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sqlite3
import threading

DEFAULT_DB_PATH = Path("reports") / "alerts_history.db"


class AlertDatabase:
    """Thread-safe SQLite storage for acoustic threat alerts."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        category TEXT NOT NULL,
                        audio_energy_dbfs REAL NOT NULL,
                        source TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active',
                        details_json TEXT DEFAULT '{}'
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts (alert_type)")
                conn.commit()

    def record_alert(
        self,
        alert_type: str,
        confidence: float,
        audio_energy_dbfs: float,
        category: str = "acoustic_threat",
        source: str = "microphone",
        timestamp: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Insert a newly triggered threat alert and return the record."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        import json
        details_str = json.dumps(details or {})

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO alerts (timestamp, alert_type, confidence, category, audio_energy_dbfs, source, status, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (timestamp, alert_type, float(confidence), category, float(audio_energy_dbfs), source, details_str),
                )
                conn.commit()
                alert_id = cursor.lastrowid

        return {
            "id": alert_id,
            "timestamp": timestamp,
            "alert_type": alert_type,
            "confidence": round(float(confidence), 4),
            "category": category,
            "audio_energy_dbfs": round(float(audio_energy_dbfs), 1),
            "source": source,
            "status": "active",
        }

    def get_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        alert_type: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical alerts with optional filtering and pagination."""
        import json
        query = "SELECT id, timestamp, alert_type, confidence, category, audio_energy_dbfs, source, status, details_json FROM alerts"
        params: List[Any] = []
        conditions = []

        if alert_type:
            conditions.append("alert_type = ?")
            params.append(alert_type)
        if search:
            conditions.append("(alert_type LIKE ? OR category LIKE ? OR source LIKE ? OR details_json LIKE ?)")
            term = f"%{search}%"
            params.extend([term, term, term, term])
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                results = []
                for row in rows:
                    details = {}
                    try:
                        raw_det = row["details_json"]
                        if raw_det:
                            details = json.loads(raw_det)
                    except Exception:
                        details = {}

                    device_name = details.get("device", row["source"])
                    results.append({
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "alert_type": row["alert_type"],
                        "confidence": round(float(row["confidence"]), 4),
                        "category": row["category"],
                        "audio_energy_dbfs": round(float(row["audio_energy_dbfs"]), 1),
                        "source": row["source"],
                        "device": device_name,
                        "status": row["status"],
                        "details": details,
                    })
                return results

    def count_alerts(
        self,
        alert_type: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Count total stored alerts, optionally by alert_type and search filters."""
        query = "SELECT COUNT(*) FROM alerts"
        params: List[Any] = []
        conditions = []

        if alert_type:
            conditions.append("alert_type = ?")
            params.append(alert_type)
        if search:
            conditions.append("(alert_type LIKE ? OR category LIKE ? OR source LIKE ? OR details_json LIKE ?)")
            term = f"%{search}%"
            params.extend([term, term, term, term])
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._lock:
            with self._get_connection() as conn:
                count = conn.execute(query, params).fetchone()[0]
                return int(count)
            params.append(alert_type)

        with self._lock:
            with self._get_connection() as conn:
                count = conn.execute(query, params).fetchone()[0]
                return int(count)

    def clear_alerts(self) -> int:
        """Clear all stored alerts (for testing/reset)."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM alerts")
                conn.commit()
                return cursor.rowcount

