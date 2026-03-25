from __future__ import annotations

import os
import re
import hmac
import hashlib
from pathlib import Path
from datetime import date, datetime

import pymysql
from flask import Flask, jsonify, request, send_file, session


BASE_DIR = Path(__file__).resolve().parent


def _resolve_data_root() -> Path:
    env_root = os.environ.get("PDMS_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    default_root = (BASE_DIR / "PDMS2").resolve()
    if default_root.exists():
        return default_root

    fallback_root = Path("/Users/yplab/Desktop/PDMS")
    if fallback_root.exists():
        return fallback_root.resolve()

    return default_root


DATA_ROOT = _resolve_data_root()
UID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
FILE_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
IMAGE_SIGN_SECRET = "pdms2-temp-sign-secret-20260325"

app = Flask(__name__, static_folder="public", static_url_path="")
app.secret_key = os.environ.get("WEB_SECRET_KEY", "dev-only-secret-change-me")

DB = dict(
    host="100.117.109.112",
    port=3306,
    user="yplab",
    password="brain0918",
    database="testPDMS",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)


def db_exec(sql: str, params=None, fetch: str = "none"):
    conn = None
    try:
        conn = pymysql.connect(**DB)
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return None
    finally:
        if conn:
            conn.close()


def is_valid_uid(uid: str) -> bool:
    return bool(UID_PATTERN.fullmatch(uid))


def is_valid_filename(filename: str) -> bool:
    return bool(FILE_PATTERN.fullmatch(filename))


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def safe_list_dir(target: Path) -> list[Path] | None:
    if not target.exists() or not target.is_dir():
        return None
    return list(target.iterdir())


def is_under_data_root(file_path: Path) -> bool:
    root = DATA_ROOT.resolve()
    candidate = file_path.resolve()
    return root == candidate or root in candidate.parents


def resolve_image_path(uid: str, filename: str) -> Path | None:
    uid_dir = (DATA_ROOT / uid).resolve()
    if not uid_dir.exists() or not uid_dir.is_dir():
        return None
    if not is_under_data_root(uid_dir):
        return None

    direct = (uid_dir / filename).resolve()
    if is_under_data_root(direct) and direct.exists() and direct.is_file():
        return direct

    lower_name = filename.lower()
    for p in uid_dir.iterdir():
        if p.is_file() and p.name.lower() == lower_name:
            if is_under_data_root(p.resolve()):
                return p.resolve()
    return None


def current_user() -> dict:
    return session.get("user") or {}


def user_level(user: dict) -> int:
    try:
        return int(user.get("level") or 0)
    except Exception:
        return 0


def user_allowed_uid(user: dict) -> str:
    return str(user.get("target_uid") or user.get("account") or "").strip()


def can_access_uid(uid: str) -> bool:
    user = current_user()
    level = user_level(user)
    if level <= 0:
        return False
    if level == 1:
        return uid == user_allowed_uid(user)
    return True


def sign_image(uid: str, filename: str) -> str:
    payload = f"{uid}/{filename}".encode("utf-8")
    return hmac.new(
        IMAGE_SIGN_SECRET.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()


def is_valid_signature(uid: str, filename: str, sig: str) -> bool:
    if not sig:
        return False
    expected = sign_image(uid, filename)
    return hmac.compare_digest(expected, sig)


def require_login() -> tuple[bool, object | None]:
    if not current_user():
        return False, (jsonify({"message": "Unauthorized"}), 401)
    return True, None


@app.post("/api/auth/login")
def api_login() -> object:
    data = request.get_json() or {}
    account = (data.get("account") or "").strip()
    password = (data.get("password") or "").strip()
    if not account or not password:
        return jsonify({"ok": False, "msg": "請輸入帳號與密碼"}), 400

    row = db_exec(
        "SELECT account, password, email, level FROM admin_users WHERE account=%s AND password=%s",
        (account, password),
        fetch="one",
    )
    if row:
        session["user"] = {
            "account": row["account"],
            "level": int(row.get("level") or 2),
            "name": row.get("email"),
            "target_uid": None,
        }
        return jsonify({"ok": True, "user": session["user"]})

    user_row = db_exec(
        "SELECT uid, name, birthday FROM user_list WHERE uid=%s",
        (account,),
        fetch="one",
    )
    if user_row:
        db_birth = user_row["birthday"]
        db_birth_str = (
            db_birth.isoformat()
            if isinstance(db_birth, (date, datetime))
            else str(db_birth or "")
        )
        if db_birth_str == password:
            session["user"] = {
                "account": user_row["uid"],
                "level": 1,
                "name": user_row["name"] or user_row["uid"],
                "target_uid": user_row["uid"],
            }
            return jsonify({"ok": True, "user": session["user"]})

    return jsonify({"ok": False, "msg": "帳號或密碼錯誤"}), 401


@app.get("/api/auth/whoami")
def api_whoami() -> object:
    user = current_user()
    if not user:
        return jsonify({"ok": True, "logged_in": False})
    return jsonify({"ok": True, "logged_in": True, "user": user})


@app.post("/api/auth/logout")
def api_logout() -> object:
    session.pop("user", None)
    return jsonify({"ok": True})


@app.get("/")
def index() -> object:
    return app.send_static_file("index.html")


@app.get("/api/uids")
def get_uids() -> object:
    ok, resp = require_login()
    if not ok:
        return resp

    user = current_user()
    if user_level(user) == 1:
        own_uid = user_allowed_uid(user)
        if not own_uid:
            return jsonify({"uids": []})
        return jsonify({"uids": [own_uid]})

    items = safe_list_dir(DATA_ROOT)
    if items is None:
        return jsonify({"message": "PDMS2 folder not found."}), 404

    uids = sorted(
        [item.name for item in items if item.is_dir() and is_valid_uid(item.name)]
    )
    return jsonify({"uids": uids})


@app.get("/api/images")
def get_images() -> object:
    ok, resp = require_login()
    if not ok:
        return resp

    uid = str(request.args.get("uid", "")).strip()
    if not is_valid_uid(uid):
        return jsonify({"message": "Invalid uid."}), 400
    if not can_access_uid(uid):
        return jsonify({"message": "Forbidden"}), 403

    uid_dir = DATA_ROOT / uid
    items = safe_list_dir(uid_dir)
    if items is None:
        return jsonify({"message": "uid folder not found."}), 404

    files = sorted(
        [
            item.name
            for item in items
            if item.is_file() and get_extension(item.name) in ALLOWED_EXTENSIONS
        ]
    )
    return jsonify({"uid": uid, "files": files})


@app.get("/images/<uid>/<filename>")
def get_image(uid: str, filename: str) -> object:
    if not is_valid_uid(uid) or not is_valid_filename(filename):
        return "Bad request.", 400

    if get_extension(filename) not in ALLOWED_EXTENSIONS:
        return "Only image files are allowed.", 400

    absolute_path = resolve_image_path(uid, filename)
    if absolute_path is None:
        return "Image not found.", 404

    signed_ok = is_valid_signature(
        uid, filename, str(request.args.get("sig", "")).strip()
    )
    session_ok = bool(current_user()) and can_access_uid(uid)
    if not signed_ok and not session_ok:
        if current_user():
            return "Forbidden", 403
        return "Unauthorized", 401

    return send_file(absolute_path)


if __name__ == "__main__":
    print(f"[MacWeb] DATA_ROOT = {DATA_ROOT}")
    app.run(host="0.0.0.0", port=3000, debug=True)
