from __future__ import annotations

import re
from pathlib import Path

from flask import Flask, jsonify, request, send_file


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "PDMS2"
UID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
FILE_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

app = Flask(__name__, static_folder="public", static_url_path="")


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


@app.get("/")
def index() -> object:
    return app.send_static_file("index.html")


@app.get("/api/uids")
def get_uids() -> object:
    items = safe_list_dir(DATA_ROOT)
    if items is None:
        return jsonify({"message": "PDMS2 folder not found."}), 404

    uids = sorted(
        [item.name for item in items if item.is_dir() and is_valid_uid(item.name)]
    )
    return jsonify({"uids": uids})


@app.get("/api/images")
def get_images() -> object:
    uid = str(request.args.get("uid", "")).strip()
    if not is_valid_uid(uid):
        return jsonify({"message": "Invalid uid."}), 400

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

    absolute_path = (DATA_ROOT / uid / filename).resolve()
    if not is_under_data_root(absolute_path):
        return "Bad request.", 400

    if not absolute_path.exists() or not absolute_path.is_file():
        return "Image not found.", 404

    return send_file(absolute_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
