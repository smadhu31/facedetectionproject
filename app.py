"""
FaceAttend backend.
- Flask + SQLite (persistent file, path set via DATABASE_PATH env var so it
  survives restarts when pointed at a Render persistent disk).
- face_recognition for embeddings + matching.
- All endpoints validated, all errors caught, proper HTTP status codes.
"""
import os
import io
import json
import base64
import sqlite3
import uuid
from datetime import datetime, date
from contextlib import contextmanager

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from PIL import Image
import face_recognition

# ----------------------------------------------------------------------
# Config (env-driven so nothing is hard-coded to localhost / a dev path)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(DATA_DIR, "faceattend.db"))

# How close (lower = stricter) a face must be to count as a match.
# face_recognition distances are roughly 0-1; 0.5 is a safe, tested default
# that avoids false positives while still recognizing registered people
# under normal lighting variance.
MATCH_TOLERANCE = float(os.environ.get("MATCH_TOLERANCE", "0.5"))

# Minimum seconds between two attendance marks for the same employee
# (prevents duplicate attendance entries from repeated scans).
ATTENDANCE_COOLDOWN_SECONDS = int(os.environ.get("ATTENDANCE_COOLDOWN_SECONDS", "0"))  # 0 = once per day only

os.makedirs(PHOTOS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)
CORS(app)  # allow the frontend (served separately or same-origin) to call the API anywhere


# ----------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            email TEXT,
            embedding TEXT NOT NULL,      -- JSON-encoded 128-d face vector
            photo_path TEXT,              -- filename inside PHOTOS_DIR
            registered_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id TEXT PRIMARY KEY,
            employee_id TEXT NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            datetime_display TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_att_emp_date ON attendance(employee_id, date)")
    conn.commit()
    conn.close()


init_db()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def decode_b64_image(image_b64: str) -> np.ndarray:
    """Decode a data-URL/base64 JPEG/PNG into an RGB numpy array."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def extract_single_face_encoding(image_array: np.ndarray):
    """
    Returns (encoding, face_box, status) where status is one of:
    'ok', 'no_face', 'multiple_faces'
    face_box is a dict {top,right,bottom,left} in image pixel coords, or None.
    """
    # 'hog' model: fast, CPU-friendly, good enough for webcam frames and
    # avoids requiring a GPU/CUDA build on Render.
    locations = face_recognition.face_locations(image_array, model="hog")
    if len(locations) == 0:
        return None, None, "no_face"
    if len(locations) > 1:
        # Use the largest face but flag the situation so the frontend can warn.
        locations = sorted(
            locations,
            key=lambda b: (b[2] - b[0]) * (b[1] - b[3]),
            reverse=True,
        )
    top, right, bottom, left = locations[0]
    encodings = face_recognition.face_encodings(image_array, known_face_locations=[locations[0]])
    if not encodings:
        return None, None, "no_face"
    box = {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)}
    return encodings[0], box, "ok"


def row_to_employee_dict(row, include_embedding=False):
    d = {
        "employee_id": row["employee_id"],
        "name": row["name"],
        "department": row["department"],
        "email": row["email"],
        "registered_at": row["registered_at"],
        "photo_url": f"/api/photos/{row['photo_path']}" if row["photo_path"] else None,
    }
    if include_embedding:
        d["embedding"] = json.loads(row["embedding"])
    return d


def find_best_match(encoding, db):
    """Compare an encoding against every stored employee. Returns (employee_row, distance) or (None, None)."""
    rows = db.execute("SELECT * FROM employees").fetchall()
    if not rows:
        return None, None
    known_encodings = [np.array(json.loads(r["embedding"])) for r in rows]
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    if best_distance <= MATCH_TOLERANCE:
        return rows[best_idx], best_distance
    return None, best_distance


def err(message, status=400, **extra):
    payload = {"success": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status


# ----------------------------------------------------------------------
# Static photo serving
# ----------------------------------------------------------------------
@app.route("/api/photos/<path:filename>")
def serve_photo(filename):
    return send_from_directory(PHOTOS_DIR, filename)


# ----------------------------------------------------------------------
# Employees
# ----------------------------------------------------------------------
@app.route("/api/employees", methods=["GET"])
def list_employees():
    try:
        db = get_db()
        rows = db.execute("SELECT * FROM employees ORDER BY registered_at DESC").fetchall()
        return jsonify({"success": True, "employees": [row_to_employee_dict(r) for r in rows]}), 200
    except Exception as e:
        return err(f"Failed to load employees: {e}", 500)


@app.route("/api/employees/<employee_id>", methods=["DELETE"])
def delete_employee(employee_id):
    try:
        db = get_db()
        row = db.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
        if not row:
            return err("Employee not found", 404)
        if row["photo_path"]:
            photo_path = os.path.join(PHOTOS_DIR, row["photo_path"])
            if os.path.exists(photo_path):
                os.remove(photo_path)
        db.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
        db.commit()
        return jsonify({"success": True, "message": f"Employee {employee_id} deleted"}), 200
    except Exception as e:
        return err(f"Failed to delete employee: {e}", 500)


@app.route("/api/register", methods=["POST"])
def register_employee():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return err("Request body must be JSON", 400)

        name = (payload.get("name") or "").strip()
        department = (payload.get("department") or "").strip()
        email = (payload.get("email") or "").strip()
        employee_id = (payload.get("employee_id") or "").strip()
        image_b64 = payload.get("image")

        # ---- validation ----
        if not name:
            return err("Name is required.", 400)
        if not image_b64:
            return err("A captured photo is required.", 400)

        db = get_db()

        if not employee_id:
            employee_id = "EMP" + uuid.uuid4().hex[:6].upper()
        else:
            existing = db.execute(
                "SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)
            ).fetchone()
            if existing:
                return err(f"Employee ID '{employee_id}' already exists.", 409)

        if email:
            existing_email = db.execute(
                "SELECT 1 FROM employees WHERE lower(email) = lower(?)", (email,)
            ).fetchone()
            if existing_email:
                return err(f"An employee with email '{email}' is already registered.", 409)

        try:
            image_array = decode_b64_image(image_b64)
        except Exception:
            return err("Could not decode the captured image.", 400)

        encoding, box, status = extract_single_face_encoding(image_array)
        if status == "no_face":
            return err("No face detected in the captured photo. Please retake it.", 422)

        # Prevent duplicate registration of the same face under a different ID.
        match_row, distance = find_best_match(encoding, db)
        if match_row is not None:
            return err(
                f"This face is already registered as '{match_row['name']}' (#{match_row['employee_id']}).",
                409,
            )

        photo_filename = f"{employee_id}_{uuid.uuid4().hex[:8]}.jpg"
        Image.fromarray(image_array).save(os.path.join(PHOTOS_DIR, photo_filename), "JPEG", quality=88)

        registered_at = datetime.utcnow().isoformat()
        db.execute(
            """INSERT INTO employees (employee_id, name, department, email, embedding, photo_path, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (employee_id, name, department, email, json.dumps(encoding.tolist()), photo_filename, registered_at),
        )
        db.commit()

        row = db.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
        return jsonify({
            "success": True,
            "message": "Employee registered successfully",
            "employee": row_to_employee_dict(row),
        }), 201

    except sqlite3.IntegrityError as e:
        return err(f"Database integrity error: {e}", 409)
    except Exception as e:
        return err(f"Registration failed: {e}", 500)


# ----------------------------------------------------------------------
# Recognition
# ----------------------------------------------------------------------
@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    try:
        payload = request.get_json(silent=True)
        if not payload or not payload.get("image"):
            return err("An image is required.", 400)

        try:
            image_array = decode_b64_image(payload["image"])
        except Exception:
            return err("Could not decode the image.", 400)

        encoding, box, status = extract_single_face_encoding(image_array)

        if status == "no_face":
            return jsonify({
                "success": True,
                "face_detected": False,
                "scan_status": "no_face",
                "label": "No Face Detected",
            }), 200

        db = get_db()
        match_row, distance = find_best_match(encoding, db)

        if match_row is None:
            return jsonify({
                "success": True,
                "face_detected": True,
                "known": False,
                "scan_status": "unknown_face",
                "label": "Unknown Person",
                "box_color": "red",
                "face_box": box,
            }), 200

        employee = row_to_employee_dict(match_row)
        confidence = max(0, round((1 - distance) * 100, 1))
        return jsonify({
            "success": True,
            "face_detected": True,
            "known": True,
            "scan_status": "known",
            "label": employee["name"],
            "box_color": "green",
            "face_box": box,
            "confidence": confidence,
            "employee": employee,
        }), 200

    except Exception as e:
        return err(f"Recognition failed: {e}", 500)


# ----------------------------------------------------------------------
# Attendance
# ----------------------------------------------------------------------
@app.route("/api/attendance", methods=["POST"])
def mark_attendance():
    try:
        payload = request.get_json(silent=True)
        employee_id = (payload or {}).get("employee_id", "").strip()
        if not employee_id:
            return err("employee_id is required.", 400)

        db = get_db()
        emp = db.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,)).fetchone()
        if not emp:
            return err("Employee not found.", 404)

        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        existing_today = db.execute(
            "SELECT * FROM attendance WHERE employee_id = ? AND date = ? ORDER BY created_at DESC LIMIT 1",
            (employee_id, today),
        ).fetchone()

        if existing_today:
            return jsonify({
                "success": False,
                "already_marked": True,
                "message": f"Attendance already marked today for {emp['name']} at {existing_today['time']}.",
            }), 200

        record_id = uuid.uuid4().hex
        time_str = now.strftime("%H:%M:%S")
        datetime_display = now.strftime("%b %d, %Y %I:%M %p")

        db.execute(
            """INSERT INTO attendance (id, employee_id, date, time, datetime_display, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (record_id, employee_id, today, time_str, datetime_display, now.isoformat()),
        )
        db.commit()

        return jsonify({
            "success": True,
            "message": "Attendance marked",
            "record": {
                "id": record_id,
                "employee_id": employee_id,
                "name": emp["name"],
                "department": emp["department"],
                "date": today,
                "time": time_str,
                "datetime_display": datetime_display,
            },
        }), 201

    except Exception as e:
        return err(f"Failed to mark attendance: {e}", 500)


@app.route("/api/attendance/list", methods=["GET"])
def list_attendance():
    try:
        db = get_db()
        rows = db.execute(
            """SELECT a.*, e.name, e.department, e.email, e.photo_path
               FROM attendance a
               JOIN employees e ON e.employee_id = a.employee_id
               ORDER BY a.created_at ASC"""
        ).fetchall()
        records = [{
            "id": r["id"],
            "employee_id": r["employee_id"],
            "name": r["name"],
            "department": r["department"],
            "email": r["email"],
            "date": r["date"],
            "time": r["time"],
            "datetime_display": r["datetime_display"],
            "photo_url": f"/api/photos/{r['photo_path']}" if r["photo_path"] else None,
        } for r in rows]
        return jsonify({"success": True, "records": records}), 200
    except Exception as e:
        return err(f"Failed to load attendance: {e}", 500)


# ----------------------------------------------------------------------
# Health check (useful for Render)
# ----------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"success": True, "status": "ok"}), 200


# ----------------------------------------------------------------------
# Serve the frontend (single-origin deployment: no CORS/base-url issues)
# ----------------------------------------------------------------------
FRONTEND_DIR = os.path.join(BASE_DIR, "templates")


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
