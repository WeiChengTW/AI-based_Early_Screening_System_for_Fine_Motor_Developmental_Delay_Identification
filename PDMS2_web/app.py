from flask import Flask, send_from_directory, request, jsonify, session
from flask_cors import CORS
import pymysql
from datetime import datetime, date
from pathlib import Path
import logging
import os
import secrets
import sys
import json
from typing import Optional
import subprocess
import traceback
import uuid
from threading import Thread
import threading
import webbrowser

# == DB == #
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


# == 建立log == #
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


# == 紀錄console == #
def write_to_console(message, level="INFO"):
    console_path = ROOT / "console.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(console_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} - {level} - {message}\n")
    except Exception as e:
        print(f"寫入 console.txt 失敗: {e}")


# == 清除console == #
def clear_console_log():
    console_path = ROOT / "console.txt"
    try:
        with open(console_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass


# == DB 執行器 == #
def db_exec(sql, params=None, fetch="none"):

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


# == 關卡代號 == #
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


# == 確認UID存在於DB== #
def user_exists(uid: str) -> bool:
    # == 回傳這個 uid 是否存在於 user_list == #
    row = db_exec(
        "SELECT 1 FROM user_list WHERE uid=%s",
        (uid,),
        fetch="one",
    )
    return row is not None


# == 確認task存在於DB == #
def task_id_to_table(task_id: str) -> str:
    if task_id in TASK_MAP:
        return TASK_MAP[task_id]
    raise ValueError(f"未知的 task_id: {task_id}")


app = Flask(
    __name__,
    static_folder=str(ROOT / "static"),
    static_url_path="/static",
)
app.secret_key = secrets.token_hex(16)
CORS(app)

clear_console_log()
app.logger.disabled = True
logging.getLogger("flask.app").disabled = True
write_to_console("== Flask Starting ! == ", "INFO")
processing_tasks = {}


# == 路由定義 == #
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


# == 設定UID (Session) == #
@app.post("/session/set-uid")
def set_session_uid():
    try:
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()
        if not uid:
            return jsonify({"success": False, "error": "UID 不能為空"}), 400
        if any(c in uid for c in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]):
            return jsonify({"success": False, "error": "UID 包含無效字符"}), 400

        #  只能用資料庫裡已存在的 UID
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
    data = request.get_json(silent=True) or {}
    uid = (data.get("uid") or "").strip()
    write_to_console(f"建立UID為{uid}的資料夾", "INFO")
    if not uid:
        write_to_console("create_uid_folder: UID 不能為空", "ERROR")
        return jsonify({"success": False, "error": "UID 不能為空"}), 400

    bad = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    if any(c in uid for c in bad):
        write_to_console(f"{uid}不存在於資料庫", "ERROR")
        return jsonify({"success": False, "error": "UID 包含無效字符"}), 400

    # == 只允許現有的UID == #
    if not user_exists(uid):
        write_to_console(f"{uid}不存在於資料庫", "WARN")
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
        write_to_console(f"建立資料夾：{kid_dir}", "INFO")

    session["uid"] = uid
    return jsonify({"success": True, "uid": uid, "message": "UID 已載入"})


@app.post("/session/clear-uid")
def clear_session_uid():
    if "uid" in session:
        del session["uid"]
    return jsonify({"success": True, "message": "UID 已清除"})


# == 分數寫入DB == #
def insert_score(
    uid: str,
    task_id: str,
    test_date: Optional[date] = None,
) -> date:

    if not user_exists(uid):
        write_to_console(f"[DB] insert_score: UID 不存在 -> {uid}", "WARN")
        raise ValueError("USER_NOT_FOUND")

    if test_date is None:
        test_date = date.today()

    current_time = datetime.now().strftime("%H:%M:%S")

    db_exec(
        """
        INSERT INTO score_list (uid, task_id, test_date)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            test_date = VALUES(test_date)
        """,
        (uid, task_id, test_date),
    )
    write_to_console(
        f"[DB] insert_score ok: uid={uid}, task_id={task_id}, date={test_date}",
        "INFO",
    )
    return test_date


# == 分析結果寫入資料庫 == #
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


# == 背景執行main.py == #
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

        # == 定義照片的實際存放路徑 == #
        # photo_path = ROOT / "kid" / uid / f"{img_id}.jpg"

        # 檢查照片是否存在，如果不存在就報錯
        # if not photo_path.exists():
        #     raise FileNotFoundError(f"找不到照片檔案: {photo_path}")

        # cmd = base_cmd + [uid, str(photo_path)]
        cmd = base_cmd + [uid, img_id]
        if stair_type:
            cmd.append(stair_type)
        # print(cmd)
        write_to_console(f"執行命令：{' '.join(cmd)}", "INFO")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        creation_flags = 0
        if sys.platform == "win32":
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

        if test_date is not None:
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

        if not img_id or not uid:
            return jsonify({"success": False, "error": "缺少參數"}), 400

        script_path = resolve_script_path(img_id)
        if not script_path or not script_path.exists():
            return jsonify({"success": False, "error": "腳本檔案不存在"}), 404

        task_id = str(uuid.uuid4())
        stair_type = session.get("stair_type")

        # print(f"UID : {uid}\nimg_id : {img_id}\ntask_id : {task_id}")
        t = Thread(
            target=run_analysis_in_background,
            args=(task_id, uid, img_id, script_path, stair_type),
        )
        t.daemon = True
        t.start()

        return jsonify({"success": True, "task_id": task_id})
    except Exception as e:
        write_to_console(f"/run-python 錯誤: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)}), 500


# == 儲存Stair Type == #
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


@app.errorhandler(Exception)
def _handle_err(e):
    tb = traceback.format_exc()
    # write_to_console(f"[ERR] {request.method} {request.path}\n{tb}", "ERROR")
    return jsonify({"success": False, "error": str(e)}), 500


def _open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}/")


# == if __name__ == "__main__" == #
if __name__ == "__main__":
    try:
        logging.getLogger("werkzeug").disabled = True
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
