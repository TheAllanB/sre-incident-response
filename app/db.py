import json
import os
import sqlite3
import uuid


def _get_conn() -> sqlite3.Connection:
    # Read DB_PATH at call time so tests can override via os.environ
    path = os.getenv("DB_PATH", "./sre_env.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def create_episode(state: dict) -> str:
    episode_id = str(uuid.uuid4())
    state = dict(state)
    state["episode_id"] = episode_id
    conn = _get_conn()
    conn.execute(
        "INSERT INTO episodes (episode_id, state_json) VALUES (?, ?)",
        (episode_id, json.dumps(state)),
    )
    conn.commit()
    conn.close()
    return episode_id


def read_episode(episode_id: str) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT state_json FROM episodes WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"Episode {episode_id} not found")
    return json.loads(row[0])


def write_episode(episode_id: str, state: dict) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE episodes SET state_json = ? WHERE episode_id = ?",
        (json.dumps(state), episode_id),
    )
    conn.commit()
    conn.close()
