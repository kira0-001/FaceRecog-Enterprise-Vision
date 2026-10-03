"""
face_db.py — SQLite database layer.
Replaces the fragile pickle registry with a proper relational database.
Two tables: people (face encodings) and attendance (recognition log).
"""

import sqlite3
import numpy as np
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "faces.db")
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


class FaceDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._attendance_cache = {}
        self._create_tables()
        print(f"[DB] SQLite database ready at: {DB_PATH}")

    def _create_tables(self):
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS people (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    UNIQUE NOT NULL,
                encoding    BLOB    NOT NULL,
                image_path  TEXT    DEFAULT '',
                created_at  TEXT    DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                seen_at     TEXT    DEFAULT (datetime('now', 'localtime')),
                confidence  REAL    DEFAULT 0.0
            );
        """)
        self.conn.commit()

    # ── PEOPLE ──────────────────────────────────────────────────────────────

    def add_person(self, name: str, encoding: np.ndarray, image_path: str = "") -> bool:
        """Insert or replace a person's face encoding. Returns True on success."""
        try:
            enc_bytes = encoding.astype(np.float64).tobytes()
            c = self.conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO people (name, encoding, image_path) VALUES (?, ?, ?)",
                (name.strip(), enc_bytes, image_path),
            )
            self.conn.commit()
            print(f"[DB] Registered: {name}")
            return True
        except Exception as e:
            print(f"[DB ERROR] add_person: {e}")
            return False

    def delete_person(self, name: str) -> bool:
        """Delete a person and their attendance records."""
        try:
            c = self.conn.cursor()
            c.execute("DELETE FROM people WHERE name=?", (name,))
            self.conn.commit()
            # Also remove their snapshot image
            img_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            if os.path.exists(img_path):
                os.remove(img_path)
            print(f"[DB] Deleted: {name}")
            return True
        except Exception as e:
            print(f"[DB ERROR] delete_person: {e}")
            return False

    def get_all_people(self) -> list:
        """Return list of dicts: name, image_path, created_at."""
        c = self.conn.cursor()
        c.execute("SELECT name, image_path, created_at FROM people ORDER BY name ASC")
        return [dict(r) for r in c.fetchall()]

    def get_registry(self) -> dict:
        """
        Returns a registry dict compatible with processing.py:
        {"names": [...], "encs": [...numpy arrays...]}
        """
        c = self.conn.cursor()
        c.execute("SELECT name, encoding FROM people")
        rows = c.fetchall()
        names, encs = [], []
        for row in rows:
            names.append(row["name"])
            encs.append(np.frombuffer(row["encoding"], dtype=np.float64).copy())
        return {"names": names, "encs": encs}

    def person_count(self) -> int:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM people")
        return c.fetchone()[0]

    # ── ATTENDANCE ───────────────────────────────────────────────────────────

    def log_attendance(self, name: str, confidence: float):
        """
        Log a recognition event. Debounces: ignores duplicate events
        for the same person within a 30-minute window to avoid spam.
        Uses in-memory cache first to avoid blocking SQLite I/O on every frame.
        """
        now = datetime.now()
        last_seen = self._attendance_cache.get(name)
        # Fast in-memory check — no disk I/O needed for 99% of frames
        if last_seen and (now - last_seen).total_seconds() < 1800:
            return

        try:
            c = self.conn.cursor()
            c.execute(
                """SELECT id FROM attendance
                   WHERE name=?
                   AND seen_at > datetime('now', 'localtime', '-30 minutes')""",
                (name,),
            )
            if c.fetchone():
                self._attendance_cache[name] = now
                return  # Debounced — already logged recently
            c.execute(
                "INSERT INTO attendance (name, confidence) VALUES (?, ?)",
                (name, round(float(confidence), 1)),
            )
            self.conn.commit()
            self._attendance_cache[name] = now
        except Exception as e:
            print(f"[DB ERROR] log_attendance: {e}")

    def get_today_attendance(self) -> list:
        """All attendance records for today, newest first."""
        c = self.conn.cursor()
        c.execute(
            """SELECT name, seen_at, confidence FROM attendance
               WHERE date(seen_at) = date('now', 'localtime')
               ORDER BY seen_at DESC"""
        )
        return [dict(r) for r in c.fetchall()]

    def clear_today_attendance(self):
        """Clear all attendance records for today."""
        c = self.conn.cursor()
        c.execute("DELETE FROM attendance WHERE date(seen_at) = date('now', 'localtime')")
        self.conn.commit()

    def get_attendance_stats(self) -> dict:
        """Summary stats for the dashboard."""
        c = self.conn.cursor()
        c.execute(
            "SELECT COUNT(*) FROM attendance WHERE date(seen_at) = date('now', 'localtime')"
        )
        total_today = c.fetchone()[0]

        c.execute(
            """SELECT COUNT(DISTINCT name) FROM attendance
               WHERE date(seen_at) = date('now', 'localtime')"""
        )
        unique_today = c.fetchone()[0]

        return {
            "total_recognitions_today": total_today,
            "unique_people_today": unique_today,
            "registered_people": self.person_count(),
        }
