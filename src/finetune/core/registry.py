"""
Model and dataset registry for tracking resources.
"""

import json
import sqlite3
from pathlib import Path

from loguru import logger


class ModelRegistry:
    """Registry for managing models."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or Path.home() / ".finetune" / "registry.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    path TEXT,
                    source TEXT,
                    size_gb REAL,
                    parameters TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    dataset_path TEXT,
                    config TEXT,
                    status TEXT,
                    metrics TEXT,
                    checkpoint_path TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """
            )

            conn.commit()

    def register_model(
        self,
        name: str,
        path: str | None = None,
        source: str = "huggingface",
        size_gb: float | None = None,
        parameters: dict | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Register a new model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO models (name, path, source, size_gb, parameters, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    path,
                    source,
                    size_gb,
                    json.dumps(parameters) if parameters else None,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            model_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Registered model: {name} (ID: {model_id})")
            return model_id

    def get_model(self, model_id: int) -> dict | None:
        """Get model by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def list_models(self) -> list[dict]:
        """List all registered models."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM models ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def start_training_run(self, model_id: int, dataset_path: str, config: dict) -> int:
        """Start a new training run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO training_runs (model_id, dataset_path, config, status)
                VALUES (?, ?, ?, ?)
                """,
                (model_id, dataset_path, json.dumps(config), "running"),
            )
            run_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Started training run: {run_id}")
            return run_id

    def update_training_run(
        self,
        run_id: int,
        status: str | None = None,
        metrics: dict | None = None,
        checkpoint_path: str | None = None,
    ):
        """Update training run status."""
        updates = []
        params = []

        if status:
            updates.append("status = ?")
            params.append(status)

        if metrics:
            updates.append("metrics = ?")
            params.append(json.dumps(metrics))

        if checkpoint_path:
            updates.append("checkpoint_path = ?")
            params.append(checkpoint_path)

        if status == "completed":
            updates.append("completed_at = CURRENT_TIMESTAMP")

        params.append(run_id)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE training_runs SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()

    def get_training_runs(self, model_id: int | None = None) -> list[dict]:
        """Get training runs, optionally filtered by model."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if model_id:
                cursor = conn.execute(
                    "SELECT * FROM training_runs WHERE model_id = ? ORDER BY started_at DESC",
                    (model_id,),
                )
            else:
                cursor = conn.execute("SELECT * FROM training_runs ORDER BY started_at DESC")
            return [dict(row) for row in cursor.fetchall()]
