# run.py (已修正：相機開啟問題 + 資料表名稱錯誤)
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import uuid
import base64
import secrets
import logging
import traceback
import subprocess
import threading
import webbrowser
from pathlib import Path
from datetime import datetime, date
from typing import Optional
from threading import Thread

import cv2
import numpy as np
import pymysql
from flask import Flask, send_from_directory, request, jsonify, session
from flask_cors import CORS

# ====== 相機參數 =====
TOP = 0
SIDE = 1  # Ch5-t1 使用
CROP_RATE = 0.8  # 預設裁切比例
# ====================

# =========================
# 1) 資料庫設定（PyMySQL 模式）
# =========================
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


def db_exec(sql, params=None, fetch="none"):
    """簡易 DB 執行器 (PyMySQL)"""
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
    except Exception as e:
        write_to_console(
            f"[DB] PyMySQL 執行失敗: {sql}\nParams: {params}\nError: {e}", "ERROR"
        )
        raise
    finally:
        if conn:
            conn.close()


TASK_MAP = {
    "Ch1-t1": "string_blocks",
    "Ch1-t2": "pyramid",
    "Ch1-t3": "stair",
    "Ch1-t4": "build_wall",
    "Ch2-t1": "draw_circle",
    "Ch2-t2": "draw_square",
    "Ch2-t3": "draw_cross",
    "Ch2-t4": "draw_line",
    "Ch2-t5": "color",
    "Ch2-t6": "connect_dots",
    "Ch3-t1": "cut_circle",
    "Ch3-t2": "cut_square",
    "Ch3-t3": "cut_paper",
    "Ch3-t4": "cut_line",
    "Ch4-t1": "one_fold",
    "Ch4-t2": "two_fold",
    "Ch5-t1": "collect_raisins",
}


# def ensure_user(uid: str, name: Optional[str] = None, birthday: Optional[str] = None):
#     try:
#         db_exec(
#             "INSERT INTO user_list(uid, name, birthday) VALUES (%s,%s,%s) "
#             "ON DUPLICATE KEY UPDATE name=COALESCE(VALUES(name),name), birthday=COALESCE(VALUES(birthday),birthday)",
#             (uid, name, birthday),
#         )
#         write_to_console(f"[DB] ensure_user ok: uid={uid}", "INFO")
#     except Exception as e:
#         raise
def user_exists(uid: str) -> bool:
    """回傳這個 uid 是否存在於 user_list"""
    row = db_exec(
        "SELECT 1 FROM user_list WHERE uid=%s",
        (uid,),
        fetch="one",  # 如果你的 db_exec 寫法不一樣，這裡用你原本查一筆資料的方式
    )
    return row is not None


def task_id_to_table(task_id: str) -> str:
    if task_id in TASK_MAP:
        return TASK_MAP[task_id]
    raise ValueError(f"未知的 task_id: {task_id}")


def insert_task_payload(
    task_id: str,
    uid: str,
    test_date: date,
    score: int,
    result_img_path: str,
    data1: Optional[str] = None,
) -> None:
    table = task_id_to_table(task_id)
    # 獲取當前時間
    current_time = datetime.now().strftime("%H:%M:%S")

    sql = f"""
        INSERT INTO `{table}` (uid, test_date, time, score, result_img_path, data1)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            score           = VALUES(score),
            result_img_path = VALUES(result_img_path),
            data1           = VALUES(data1),
            time            = VALUES(time)
    """
    try:
        db_exec(sql, (uid, test_date, current_time, score, result_img_path, data1))
    except Exception:
        raise


def ensure_task(task_id: str):
    if task_id not in TASK_MAP:
        raise ValueError(f"未知的 task_id：{task_id}")
    task_name = TASK_MAP[task_id]
    try:
        db_exec(
            "INSERT INTO task_list(task_id, task_name) VALUES (%s,%s) "
            "ON DUPLICATE KEY UPDATE task_name=VALUES(task_name)",
            (task_id, task_name),
        )
        write_to_console(f"[DB] ensure_task ok: {task_id} -> {task_name}", "INFO")
    except Exception as e:
        raise


def insert_score(
    uid: str,
    task_id: str,
    test_date: Optional[date] = None,
) -> date:

    if not user_exists(uid):
        write_to_console(f"[DB] insert_score: UID 不存在 -> {uid}", "WARN")
        # 丟一個明確的錯誤，讓呼叫的人去決定要怎麼回應前端
        raise ValueError("USER_NOT_FOUND")

    # task 邏輯照舊（如果你希望只有管理者能新增 task，也可以之後再改 ensure_task）
    ensure_task(task_id)

    if test_date is None:
        test_date = date.today()

    current_time = datetime.now().strftime("%H:%M:%S")

    db_exec(
        """
        INSERT INTO score_list (uid, task_id, test_date, time)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            test_date = VALUES(test_date),
            time = VALUES(time)
        """,
        (uid, task_id, test_date, current_time),
    )
    write_to_console(
        f"[DB] insert_score ok: uid={uid}, task_id={task_id}, date={test_date}, time={current_time}",
        "INFO",
    )
    return test_date


# =========================
# 2) 基礎環境/日誌/靜態路由
# =========================
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# ====== MAC 上傳設定（直接寫在程式，不使用 .env） ======
MAC_UPLOAD_HOST = "100.117.109.112"
MAC_UPLOAD_PORT = 22
MAC_UPLOAD_USER = "YPLAB"
MAC_UPLOAD_PASSWORD = "brain0918"
MAC_UPLOAD_KEY_PATH = ""
MAC_UPLOAD_REMOTE_BASE = "Desktop/PDMS"
# =====================================================

PORT = 8000
HOST = "127.0.0.1"
ROOT = Path(__file__).parent.resolve()

app = Flask(
    __name__,
    static_folder=str(ROOT / "static"),
    static_url_path="/static",
)
app.secret_key = secrets.token_hex(16)
CORS(app)


def setup_console_logging():
    console_path = Path(__file__).parent / "console.txt"
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(console_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    lg = logging.getLogger("app")
    lg.setLevel(logging.INFO)
    lg.addHandler(fh)
    lg.propagate = False
    return lg


def write_to_console(message, level="INFO"):
    console_path = ROOT / "console.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(console_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} - {level} - {message}\n")
    except Exception as e:
        print(f"寫入 console.txt 失敗: {e}")


def clear_console_log():
    console_path = ROOT / "console.txt"
    try:
        with open(console_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass


clear_console_log()
logger = setup_console_logging()
app.logger.disabled = True
logging.getLogger("flask.app").disabled = True
write_to_console("=== 遠端 PyMySQL 模式 ===", "INFO")
write_to_console("Flask 應用程式啟動", "INFO")
processing_tasks = {}


@app.route("/")
def home():
    return send_from_directory(ROOT / "html", "start.html")


@app.route("/index")
@app.route("/index.html")
def index_shortcut():
    return send_from_directory(ROOT / "html", "index.html")


@app.route("/html/<path:filename>")
def html_files(filename):
    return send_from_directory(ROOT / "html", filename)


@app.route("/css/<path:filename>")
def css_files(filename):
    return send_from_directory(ROOT / "css", filename)


@app.route("/js/<path:filename>")
def js_files(filename):
    return send_from_directory(ROOT / "js", filename)


@app.route("/images/<path:filename>")
def images_files(filename):
    return send_from_directory(ROOT / "images", filename)


@app.route("/kid/<path:filename>")
def kid_files(filename):
    # 讓網頁可以讀取 kid 資料夾內的照片
    return send_from_directory(ROOT / "kid", filename)


@app.route("/video/<path:filename>")
def video_files(filename):
    return send_from_directory(ROOT / "video", filename)


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    return ("", 204)


@app.get("/logs/tail")
def logs_tail():
    try:
        n = int(request.args.get("n", 200))
        p = ROOT / "console.txt"
        if not p.exists():
            return jsonify({"ok": False, "msg": "console.txt not found"}), 404
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return jsonify({"ok": True, "lines": lines})
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500


@app.before_request
def _log_request():
    if request.path.startswith(
        ("/css/", "/js/", "/images/", "/video/", "/favicon.ico", "/opencv-camera/")
    ):
        return
    try:
        if request.path.startswith(
            ("/run-python", "/create-uid-folder", "/test-score")
        ):
            write_to_console(f"[REQ] {request.method} {request.path}")
    except Exception as e:
        write_to_console(f"[REQ] log failed: {e}", "ERROR")


@app.after_request
def _log_response(resp):
    if request.path.startswith(
        ("/css/", "/js/", "/images/", "/video/", "/favicon.ico", "/opencv-camera/")
    ):
        return resp
    try:
        if resp.status_code >= 400 or request.path.startswith(
            ("/run-python", "/create-uid-folder", "/test-score")
        ):
            write_to_console(
                f"[RESP] {request.method} {request.path} -> {resp.status_code}"
            )
    except Exception as e:
        write_to_console(f"[RESP] log failed: {e}", "ERROR")
    return resp


@app.errorhandler(Exception)
def _handle_err(e):
    tb = traceback.format_exc()
    write_to_console(f"[ERR] {request.method} {request.path}\n{tb}", "ERROR")
    return jsonify({"success": False, "error": str(e)}), 500


def _open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}/")


# =========================
# 3) Session：UID
# =========================
@app.post("/session/set-uid")
def set_session_uid():
    try:
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()
        if not uid:
            return jsonify({"success": False, "error": "UID 不能為空"}), 400
        if any(c in uid for c in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]):
            return jsonify({"success": False, "error": "UID 包含無效字符"}), 400

        # ⭐ 新增：只能用資料庫裡已存在的 UID
        if not user_exists(uid):
            write_to_console(f"set_session_uid: UID 不存在 -> {uid}", "WARN")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "此使用者不存在，請請管理者建立帳號",
                        "code": "USER_NOT_FOUND",
                    }
                ),
                404,
            )

        session["uid"] = uid
        write_to_console(f"成功設置 UID：{uid}", "INFO")
        return jsonify({"success": True, "uid": uid})
    except Exception as e:
        write_to_console(f"設置 UID 時發生錯誤：{e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/session/get-uid", methods=["GET"])
def get_session_uid():
    uid = session.get("uid")
    return jsonify({"success": True, "uid": uid})


@app.post("/create-uid-folder")
def create_uid_folder():
    write_to_console("[REQ] 進入 create_uid_folder", "INFO")
    data = request.get_json(silent=True) or {}
    uid = (data.get("uid") or "").strip()
    if not uid:
        write_to_console("create_uid_folder: UID 不能為空", "ERROR")
        return jsonify({"success": False, "error": "UID 不能為空"}), 400

    bad = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    if any(c in uid for c in bad):
        write_to_console(f"create_uid_folder: UID 非法 -> {uid}", "ERROR")
        return jsonify({"success": False, "error": "UID 包含無效字符"}), 400

    # ⭐ 不再自動新增，只允許已存在的 UID
    if not user_exists(uid):
        write_to_console(f"create_uid_folder: UID 不存在 -> {uid}", "WARN")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "此使用者不存在，請請管理者建立帳號",
                    "code": "USER_NOT_FOUND",
                }
            ),
            404,
        )

    kid_dir = ROOT / "kid" / uid
    if not kid_dir.exists():
        kid_dir.mkdir(parents=True, exist_ok=True)
        write_to_console(f"[FS] 建立資料夾：{kid_dir}", "INFO")

    session["uid"] = uid
    return jsonify({"success": True, "uid": uid, "message": "UID 已載入"})


@app.post("/session/clear-uid")
def clear_session_uid():
    if "uid" in session:
        del session["uid"]
    return jsonify({"success": True, "message": "UID 已清除"})


@app.route("/test-score", methods=["POST"])
def test_score():
    try:
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()
        task_id = (data.get("task_id") or "").strip()

        if not uid or not task_id:
            return jsonify({"success": False, "error": "uid 與 task_id 不可為空"}), 400

        score = 3  # 你目前先寫死 3 分

        try:
            # 這裡可能會因為 UID 不存在而丟 ValueError("USER_NOT_FOUND")
            test_date = insert_score(uid=uid, task_id=task_id)
        except ValueError as e:
            if str(e) == "USER_NOT_FOUND":
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "此使用者不存在，請管理者建立帳號",
                            "code": "USER_NOT_FOUND",
                        }
                    ),
                    404,
                )
            # 其他 ValueError 再往上丟，交給外層 except
            raise

        insert_task_payload(
            task_id=task_id,
            uid=uid,
            test_date=test_date,
            score=score,
            result_img_path="",
            data1=None,
        )
        return jsonify(
            {
                "success": True,
                "uid": uid,
                "task_id": task_id,
                "test_date": test_date.isoformat(),
                "score": score,
            }
        )
    except Exception as e:
        write_to_console(f"/test-score 錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# 4) 背景執行 main.py
# =========================
def normalize_task_id(task_code_raw: str) -> str:
    if task_code_raw in TASK_MAP:
        return task_code_raw
    parts = task_code_raw.split("-")
    if len(parts) == 2:
        guess = parts[0][:1].upper() + parts[0][1:] + "-" + parts[1]
        if guess in TASK_MAP:
            return guess
    return task_code_raw


def resolve_script_path(task_code: str) -> Optional[Path]:
    guesses = [
        ROOT / task_code / "main.py",
        ROOT / task_code.lower() / "main.py",
        ROOT / normalize_task_id(task_code) / "main.py",
        ROOT / normalize_task_id(task_code).lower() / "main.py",
    ]
    for p in guesses:
        if p.exists():
            return p
    return None


def upload_task_images_to_mac(uid: str, img_id: str) -> bool:
    uploader_path = ROOT / "scripts" / "upload_to_mac_pdms.py"
    if not uploader_path.exists():
        write_to_console(f"[UPLOAD] 找不到上傳腳本: {uploader_path}", "ERROR")
        return False

    if not MAC_UPLOAD_HOST or not MAC_UPLOAD_USER:
        write_to_console(
            "[UPLOAD] 缺少上傳設定：MAC_UPLOAD_HOST 或 MAC_UPLOAD_USER", "ERROR"
        )
        return False

    cmd = [
        sys.executable,
        str(uploader_path),
        "--uid",
        uid,
        "--img-id",
        img_id,
        "--root",
        str(ROOT),
        "--host",
        MAC_UPLOAD_HOST,
        "--port",
        str(MAC_UPLOAD_PORT),
        "--user",
        MAC_UPLOAD_USER,
        "--remote-base",
        MAC_UPLOAD_REMOTE_BASE,
    ]

    if MAC_UPLOAD_PASSWORD:
        cmd.extend(["--password", MAC_UPLOAD_PASSWORD])
    if MAC_UPLOAD_KEY_PATH:
        cmd.extend(["--key-path", MAC_UPLOAD_KEY_PATH])

    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags,
    )

    if result.stdout:
        write_to_console(f"[UPLOAD][stdout]\n{result.stdout}", "INFO")
    if result.stderr:
        write_to_console(f"[UPLOAD][stderr]\n{result.stderr}", "ERROR")

    if result.returncode == 0:
        write_to_console(f"[UPLOAD] 上傳成功: uid={uid}, img_id={img_id}", "INFO")
        return True

    write_to_console(
        f"[UPLOAD] 上傳失敗(returncode={result.returncode}): uid={uid}, img_id={img_id}",
        "ERROR",
    )
    return False


def run_analysis_in_background(
    task_id, uid, img_id, script_path, stair_type=None, cam_index_input=None
):
    try:
        processing_tasks[task_id] = {
            "status": "running",
            "uid": uid,
            "img_id": img_id,
            "start_time": datetime.now().isoformat(),
            "progress": 0,
        }
        write_to_console(f"開始背景任務 {task_id}: uid={uid}, task={img_id}", "INFO")

        base_cmd = [sys.executable, str(script_path)]
        is_game = normalize_task_id(img_id) == "Ch5-t1"
        camera_to_use = SIDE
        if is_game and cam_index_input is not None:
            try:
                camera_to_use = int(cam_index_input)
            except ValueError:
                pass

        if is_game:
            release_camera()
            time.sleep(1.5)
            cmd = base_cmd + [uid, str(camera_to_use)]
        else:
            cmd = base_cmd + [uid, img_id]
            if stair_type:
                cmd.append(stair_type)

        write_to_console(f"執行命令：{' '.join(cmd)}", "INFO")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        creation_flags = 0
        if sys.platform == "win32" and not is_game:
            creation_flags = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )

        score = int(result.returncode)
        stdout_str = result.stdout if result.stdout else ""
        stderr_str = result.stderr if result.stderr else ""

        if stdout_str:
            write_to_console(f"腳本輸出 (任務 {task_id})：\n{stdout_str}", "INFO")
        if stderr_str:
            write_to_console(f"腳本錯誤輸出 (任務 {task_id})：\n{stderr_str}", "ERROR")

        task_id_std = normalize_task_id(img_id)
        uid_eff = uid or "unknown"
        test_date = None

        try:
            test_date = insert_score(uid_eff, task_id_std)
        except Exception as e:
            write_to_console(f"寫入 score_list 失敗：{e}", "ERROR")

        if (test_date is not None) and (not is_game):
            try:
                current_img_path = f"kid/{uid_eff}/{img_id}.jpg"
                insert_task_payload(
                    task_id=task_id_std,
                    uid=uid_eff,
                    test_date=test_date,
                    score=score,
                    result_img_path=current_img_path,
                    data1=None,
                )
            except Exception as e:
                write_to_console(f"寫入任務子表失敗: {e}", "ERROR")

        try:
            write_to_console(
                f"[UPLOAD] 任務完成後開始上傳: task={img_id}, uid={uid_eff}",
                "INFO",
            )
            upload_ok = upload_task_images_to_mac(uid_eff, img_id)
            write_to_console(
                f"[UPLOAD] 任務上傳結果: task={img_id}, uid={uid_eff}, success={upload_ok}",
                "INFO" if upload_ok else "ERROR",
            )
        except Exception as e:
            write_to_console(f"[UPLOAD] 執行上傳時發生例外: {e}", "ERROR")

        processing_tasks[task_id] = {
            "status": "completed",
            "uid": uid_eff,
            "img_id": img_id,
            "end_time": datetime.now().isoformat(),
            "progress": 100,
            "result": {"success": True, "returncode": score, "task_id": task_id_std},
        }
        write_to_console(f"任務 {task_id} 完成：score={score}", "INFO")

    except Exception as e:
        tb = traceback.format_exc()
        processing_tasks[task_id] = {"status": "error", "error": str(e)}
        write_to_console(f"背景任務 {task_id} 錯誤：{e}\n{tb}", "ERROR")


@app.post("/run-python")
def run_python_script():
    try:
        data = request.get_json() or {}
        img_id = (data.get("id") or "").strip()
        uid = (data.get("uid") or "").strip() or session.get("uid")
        cam_index_input = data.get("cam_index")

        if not img_id or not uid:
            return jsonify({"success": False, "error": "缺少參數"}), 400

        script_path = resolve_script_path(img_id)
        if not script_path or not script_path.exists():
            return jsonify({"success": False, "error": "腳本檔案不存在"}), 404

        task_id = str(uuid.uuid4())
        stair_type = session.get("stair_type")
        processing_tasks[task_id] = {"status": "pending", "uid": uid}

        t = Thread(
            target=run_analysis_in_background,
            args=(task_id, uid, img_id, script_path, stair_type, cam_index_input),
        )
        t.daemon = True
        t.start()

        return jsonify({"success": True, "task_id": task_id})
    except Exception as e:
        write_to_console(f"/run-python 錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/check-task/<task_id>")
def check_task_status(task_id):
    if task_id not in processing_tasks:
        return jsonify({"success": False, "error": "任務不存在"}), 404
    return jsonify({"success": True, **processing_tasks[task_id], "task_id": task_id})


@app.post("/save-stair-type")
def save_stair_type():
    try:
        data = request.get_json() or {}
        stair_type = data.get("stair_type", "").strip()
        if stair_type not in ["L", "R"]:
            return jsonify({"success": False, "error": "stair_type 只能是 L 或 R"}), 400
        session["stair_type"] = stair_type
        return jsonify({"success": True, "stair_type": stair_type})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# 5) 便利檢查 API
# =========================
@app.get("/db/ping")
def db_ping():
    try:
        v = db_exec("SELECT VERSION() AS v", fetch="one")
        return jsonify({"ok": True, "version": v["v"] if v else "?"})
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500


# ★★★★★★★ 新增：搜尋成績 API (修復 admin.js 初始化失敗) ★★★★★★★
@app.post("/api/search-scores")
def search_scores_api():
    try:
        data = request.get_json() or {}
        uid = data.get("uid", "").strip()
        task_id = data.get("task_id", "").strip()
        rows = []

        # 情況 1: 指定關卡 -> 查該關卡的資料表
        if task_id:
            try:
                table = task_id_to_table(task_id)
                sql = f"SELECT uid, '{task_id}' as task_id, score, test_date, time, result_img_path FROM `{table}` WHERE uid=%s ORDER BY test_date DESC, time DESC"
                rows = db_exec(sql, (uid,), fetch="all")
            except Exception as e:
                write_to_console(f"查詢子表失敗 {task_id}: {e}", "WARN")
                rows = []

        # 情況 2: 沒指定關卡 -> 查總表 score_list 並補上路徑
        if not rows and not task_id:
            sql = "SELECT uid, task_id, score, test_date, time FROM score_list WHERE uid=%s ORDER BY test_date DESC, time DESC"
            base_rows = db_exec(sql, (uid,), fetch="all")
            if base_rows:
                for r in base_rows:
                    # 預設路徑格式：kid/UID/關卡名.jpg
                    r["result_img_path"] = f"kid/{r['uid']}/{r['task_id'].lower()}.jpg"
                    rows.append(r)

        # 格式化日期
        for r in rows:
            if isinstance(r.get("test_date"), (date, datetime)):
                r["test_date"] = r["test_date"].isoformat()
            if r.get("time") and not isinstance(r.get("time"), str):
                r["time"] = str(r["time"])

        return jsonify({"success": True, "data": rows})
    except Exception as e:
        write_to_console(f"搜尋失敗: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# =========================
# 6) OpenCV 相機
# =========================
camera = None
camera_active = False


def release_camera():
    global camera, camera_active
    if camera is not None:
        try:
            camera.release()
            write_to_console("[相機] 相機已釋放", "INFO")
        except Exception as e:
            write_to_console(f"[相機] 釋放失敗: {e}", "WARN")
        camera = None
    camera_active = False
    time.sleep(0.3)


# ★★★★★ 修正 2：相機初始化改用 DSHOW + 自動切換 ★★★★★
def init_camera(camera_index=TOP):
    global camera, camera_active
    try:
        release_camera()

        # 優先嘗試 cv2.CAP_DSHOW (Windows 推薦)
        print(f"[相機] 嘗試開啟相機 Index: {camera_index} (DSHOW)...")
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        # 如果打不開，嘗試不指定後端 (自動)
        if not camera.isOpened():
            print(f"[相機] DSHOW 失敗，嘗試自動後端開啟 Index: {camera_index}...")
            camera = cv2.VideoCapture(camera_index)

        # 如果還是打不開，且原本不是 0，嘗試強制切回 0 (預設)
        if not camera.isOpened() and camera_index != 0:
            print("[相機] 指定鏡頭失敗，嘗試切換回預設鏡頭 (Index 0)...")
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            raise Exception(f"無法開啟任何相機 (Index: {camera_index})")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)

        # 讀取測試
        ret, frame = camera.read()
        if not ret:
            # 有時候剛開啟第一幀會是黑的，再試一次
            time.sleep(0.5)
            ret, frame = camera.read()
            if not ret:
                raise Exception("相機已開啟但無法讀取畫面")

        h, w = frame.shape[:2]
        crop_w = int(w * CROP_RATE)
        crop_h = int(h * CROP_RATE)
        write_to_console(
            f"相機開啟成功，來源: {camera_index}，原始尺寸: {w}x{h}，裁切後: {crop_w}x{crop_h}",
            "INFO",
        )
        camera_active = True
        return True

    except Exception as e:
        write_to_console(f"相機初始化失敗: {e}", "ERROR")
        release_camera()
        return False


def crop_center(frame, rate):
    h, w = frame.shape[:2]
    crop_w = int(w * rate)
    crop_h = int(h * rate)
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    return frame[start_y : start_y + crop_h, start_x : start_x + crop_w]


def get_frame():
    global camera, camera_active
    if not camera_active or camera is None:
        return None
    try:
        ret, frame = camera.read()
        if not ret:
            return None
        frame = crop_center(frame, CROP_RATE)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    except Exception:
        return None


@app.post("/opencv-camera/stop")
def stop_opencv_camera():
    release_camera()
    return jsonify({"success": True})


@app.post("/opencv-camera/start")
def start_opencv_camera():
    try:
        data = request.get_json() or {}
        cam_index = data.get("camera_index", TOP)
        if init_camera(cam_index):
            return jsonify({"success": True})
        return jsonify({"success": False}), 500
    except:
        return jsonify({"success": False}), 500


@app.get("/opencv-camera/frame")
def get_opencv_frame():
    if not camera_active:
        return jsonify({"success": False}), 400
    frame_data = get_frame()
    if frame_data is None:
        return jsonify({"success": False}), 500
    img_b64 = base64.b64encode(frame_data).decode("utf-8")
    return jsonify({"success": True, "image": img_b64})


@app.post("/opencv-camera/capture")
def capture_opencv_photo():
    try:
        data = request.get_json() or {}
        task_id_input = (data.get("task_id") or "").strip()
        uid = (data.get("uid") or "").strip() or session.get("uid", "default")

        if not task_id_input:
            return jsonify({"success": False}), 400

        frame_data = get_frame()
        if frame_data is None:
            return jsonify({"success": False}), 500

        target_dir = ROOT / "kid" / uid
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{task_id_input}.jpg"
        file_path = target_dir / filename

        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(str(file_path), img)
        write_to_console(f"圖像存儲成功: {file_path}", "INFO")

        # 修正：不自動啟動分析，交由前端呼叫 /run-python
        return jsonify(
            {
                "success": True,
                "uid": uid,
                "task_id": task_id_input,
                "filename": filename,
                "analysis_started": False,
            }
        )
    except Exception:
        return jsonify({"success": False}), 500


@app.get("/game-state/<uid>")
def get_game_state(uid):
    try:
        state_file = ROOT / "kid" / uid / "Ch5-t1_state.json"
        if not state_file.exists():
            return jsonify({"success": False, "error": "狀態檔案不存在"}), 404
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        return jsonify({"success": True, "state": state})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/clear-game-state")
def clear_game_state():
    try:
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()
        if not uid:
            return jsonify({"success": False}), 400

        state_file = ROOT / "kid" / uid / "Ch5-t1_state.json"
        initial_state = {
            "running": False,
            "bean_count": 0,
            "remaining_time": 60,
            "warning": False,
            "game_over": False,
            "score": -1,
        }
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(initial_state, f, ensure_ascii=False, indent=2)
        write_to_console(f"[Ch5-t1] 遊戲狀態已清空: {uid}", "INFO")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    try:
        write_to_console(f"準備啟動 Flask 應用程式，HOST={HOST}, PORT={PORT}", "INFO")
        threading.Timer(0.5, _open_browser).start()
        write_to_console("已設定瀏覽器自動開啟", "INFO")
    except Exception as e:
        write_to_console(f"設定瀏覽器自動開啟時發生錯誤：{str(e)}", "ERROR")

    try:
        write_to_console("Flask 應用程式開始運行", "INFO")
        cli = sys.modules["flask.cli"]
        cli.show_server_banner = lambda *x: None
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        write_to_console("接收到中斷信號，正在關閉應用程式", "INFO")
    except Exception as e:
        write_to_console(f"應用程式運行時發生錯誤：{str(e)}", "ERROR")
    finally:
        write_to_console("Flask 應用程式已關閉", "INFO")
