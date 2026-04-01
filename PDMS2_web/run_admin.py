# run_admin.py (權限分離修正版)
# -*- coding: utf-8 -*-
import os
import sys
import json
import secrets
import threading
import webbrowser
from pathlib import Path
from datetime import datetime, date

import pymysql
from flask import Flask, send_from_directory, request, jsonify, session
from flask_cors import CORS

# =========================
# 1) 資料庫設定
# =========================
DB = dict(
    host="13.238.239.23",
    port=3306,
    user="project",
    password="project",
    database="pdms2",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

def db_exec(sql, params=None, fetch="none"):
    conn = None
    try:
        conn = pymysql.connect(**DB)
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if fetch == "one": return cur.fetchone()
            if fetch == "all": return cur.fetchall()
            return None
    except Exception as e:
        print(f"[DB Error] {e}") 
        raise
    finally:
        if conn: conn.close()

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

MULTI_ANGLE_TASKS = ["Ch1-t2", "Ch1-t3", "Ch1-t4"]

def normalize_task_id(tid: str) -> str:
    if tid in TASK_MAP: return tid
    tid_lower = tid.lower()
    for k in TASK_MAP.keys():
        if k.lower() == tid_lower:
            return k
    return tid

def task_id_to_table(task_id: str) -> str:
    clean_id = normalize_task_id(task_id)
    if clean_id in TASK_MAP: return TASK_MAP[clean_id]
    raise ValueError(f"未知的 task_id: {task_id}")

def ensure_user(uid: str):
    try: db_exec("INSERT INTO user_list(uid) VALUES (%s) ON DUPLICATE KEY UPDATE uid=uid", (uid,))
    except: pass 

def ensure_task(task_id: str):
    clean_id = normalize_task_id(task_id)
    if clean_id not in TASK_MAP: raise ValueError(f"未知的 task_id：{task_id}")
    task_name = TASK_MAP[clean_id]
    try: db_exec("INSERT INTO task_list(task_id, task_name) VALUES (%s,%s) ON DUPLICATE KEY UPDATE task_name=VALUES(task_name)", (clean_id, task_name))
    except: pass

def make_row_key(uid, task_id, test_date_str: str, time_str: str = ""):
    if time_str: return f"{uid}|{task_id}|{test_date_str}|{time_str}"
    return f"{uid}|{task_id}|{test_date_str}"

def _rows_date_to_str(rows):
    out = []
    for r in rows or []:
        r = dict(r)
        td = r.get("test_date")
        if isinstance(td, (date, datetime)): r["test_date"] = td.isoformat()
        if "time" in r and not isinstance(r["time"], str): r["time"] = str(r["time"])
        out.append(r)
    return out

# =========================
# 2) Flask 設定
# =========================
os.environ["PYTHONIOENCODING"] = "utf-8"
PORT = 8001
HOST = "127.0.0.1"
ROOT = Path(__file__).parent.resolve()

app = Flask(__name__, static_folder=None)
app.secret_key = secrets.token_hex(16)
CORS(app, supports_credentials=True)

def _open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}/")

# =========================
# 3) 靜態檔案與路由
# =========================
@app.route("/")
def home(): return send_from_directory(ROOT / "html", "admin_login.html")

@app.route("/admin")
@app.route("/admin.html")
def admin_page(): return send_from_directory(ROOT / "html", "admin.html")

@app.route("/html/<path:filename>")
def html_files(filename): return send_from_directory(ROOT / "html", filename)

@app.route("/css/<path:filename>")
def css_files(filename): return send_from_directory(ROOT / "css", filename)

@app.route("/js/<path:filename>")
def js_files(filename): return send_from_directory(ROOT / "js", filename)

@app.route("/images/<path:filename>")
def images_files(filename): return send_from_directory(ROOT / "images", filename)

@app.route("/kid/<path:relpath>")
def kid(relpath):
    relpath = relpath.replace("\\", "/").lstrip("/")
    if ".." in relpath: return ("", 404)
    if relpath.startswith("kid/"): relpath = relpath[4:]
    abs_path = ROOT / "kid" / relpath
    if not abs_path.exists(): return ("", 404)
    return send_from_directory(abs_path.parent, abs_path.name)

@app.route("/session/get-uid")
def get_session_uid_shim():
    user = session.get("user")
    uid = user.get("account") if user else None
    return jsonify({"success": True, "uid": uid})

@app.route("/view-compare")
def view_compare():
    uid = request.args.get("uid", "")
    task_id = request.args.get("task_id", "")
    img_path = request.args.get("img", "") 

    if not uid or not task_id: return "Missing uid or task_id", 400
    task_id = normalize_task_id(task_id)
    is_multi = task_id in MULTI_ANGLE_TASKS
    
    content_html = ""
    if is_multi:
        side_orig = f"/kid/{uid}/{task_id}-side.jpg"
        side_res  = f"/kid/{uid}/{task_id}-side_result.jpg"
        top_orig  = f"/kid/{uid}/{task_id}-top.jpg"
        top_res   = f"/kid/{uid}/{task_id}-top_result.jpg"

        content_html = f"""
        <div class="section-title">側面視角 (Side View)</div>
        <div class="row">
            <div class="box">
                <h3>原始照片</h3>
                <img src="{side_orig}" onerror="this.onerror=null;this.src='/images/no_image.png';">
            </div>
            <div class="box">
                <h3>分析結果</h3>
                <img src="{side_res}" onerror="this.onerror=null;this.src='/images/no_image.png';">
            </div>
        </div>
        <div class="section-title" style="margin-top: 40px; border-top: 2px dashed #ddd; padding-top:20px;">頂部視角 (Top View)</div>
        <div class="row">
            <div class="box">
                <h3>原始照片</h3>
                <img src="{top_orig}" onerror="this.onerror=null;this.src='/images/no_image.png';">
            </div>
            <div class="box">
                <h3>分析結果</h3>
                <img src="{top_res}" onerror="this.onerror=null;this.src='/images/no_image.png';">
            </div>
        </div>
        """
    else:
        original_src = img_path if img_path else f"/kid/{uid}/{task_id}.jpg"
        result_src = f"/kid/{uid}/{task_id}_result.jpg"
        if "_result" in original_src: 
            result_src = original_src
            original_src = original_src.replace("_result.jpg", ".jpg")
        elif ".jpg" in original_src:
            result_src = original_src.replace(".jpg", "_result.jpg")

        content_html = f"""
        <div class="row">
            <div class="box">
                <h3>原始照片 (Original)</h3>
                <img src="{original_src}" onerror="this.onerror=null;this.src='/images/no_image.png';">
            </div>
            <div class="box">
                <h3>分析結果 (Result)</h3>
                <img src="{result_src}" onerror="this.onerror=null;this.src='/images/no_image.png';">
            </div>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <title>作答結果比對 - {uid} - {task_id}</title>
        <style>
            body {{ font-family: "Microsoft JhengHei", sans-serif; text-align: center; padding: 20px; background: #f0f2f5; }}
            h2 {{ color: #333; margin-bottom: 10px; }}
            .sub-info {{ color: #666; margin-bottom: 30px; font-size: 0.9em; }}
            .row {{ display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
            .box {{ 
                background: white; padding: 10px; border-radius: 8px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                width: 45%; min-width: 300px;
            }}
            .box h3 {{ margin: 0 0 10px 0; color: #555; font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #eee; }}
            .section-title {{ 
                font-size: 18px; font-weight: bold; color: #2c3e50; 
                margin: 10px 0; display: inline-block; background: #e0f2fe; 
                padding: 5px 15px; border-radius: 20px; 
            }}
        </style>
    </head>
    <body>
        <h2>使用者: {uid} / 關卡: {task_id}</h2>
        <div class="sub-info">檢視模式: {"多視角" if is_multi else "單一視角"}</div>
        {content_html}
    </body>
    </html>
    """
    return html

# =========================
# 4) 核心 API
# =========================

@app.post("/api/auth/login")
def api_login():
    data = request.get_json() or {}
    account = (data.get("account") or "").strip()
    password = (data.get("password") or "").strip()
    if not account or not password: return jsonify({"ok": False, "msg": "請輸入帳號與密碼"}), 400

    try:
        row = db_exec("SELECT account, password, email, level FROM admin_users WHERE account=%s AND password=%s", (account, password), fetch="one")
        if row:
            # 這裡 DB 的 level 如果是 3，就會存入 session
            session["user"] = {"account": row["account"], "level": int(row.get("level") or 2), "name": row.get("email"), "target_uid": None}
            return jsonify({"ok": True, "user": session["user"]})
    except Exception as e: print(f"[Admin Login Check] {e}")

    try:
        user_row = db_exec("SELECT uid, name, birthday FROM user_list WHERE uid=%s", (account,), fetch="one")
        if user_row:
            db_birth = user_row["birthday"]
            db_birth_str = db_birth.isoformat() if isinstance(db_birth, (date, datetime)) else str(db_birth or "")
            if db_birth_str == password:
                session["user"] = {"account": user_row["uid"], "level": 1, "name": user_row["name"] or user_row["uid"], "target_uid": user_row["uid"]}
                return jsonify({"ok": True, "user": session["user"]})
    except Exception as e: print(f"[Parent Login Check] {e}")

    return jsonify({"ok": False, "msg": "帳號或密碼錯誤"}), 401

@app.get("/api/auth/whoami")
def api_whoami():
    user = session.get("user")
    if not user: return jsonify({"ok": True, "logged_in": False})
    return jsonify({"ok": True, "logged_in": True, "user": user})

@app.post("/api/auth/logout")
def api_logout():
    session.pop("user", None)
    return jsonify({"ok": True})

@app.get("/users")
def list_users():
    try:
        user = session.get("user") or {}
        # Level 2 和 3 都可以看
        if int(user.get("level") or 0) < 2: return jsonify({"ok": False, "msg": "權限不足"}), 403

        sql = """
            SELECT uid, 
                   CASE 
                     WHEN name IS NOT NULL AND CHAR_LENGTH(name) > 0 THEN CONCAT(name, ' (', uid, ')')
                     ELSE uid 
                   END AS name
            FROM user_list 
            ORDER BY uid ASC
        """
        rows = db_exec(sql, fetch="all")
        return jsonify({"ok": True, "users": rows})
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500

@app.get("/tasks")
def list_tasks():
    try:
        rows = db_exec("SELECT task_id FROM task_list ORDER BY task_id", fetch="all")
        tasks = [r["task_id"] for r in (rows or [])]
        return jsonify({"ok": True, "tasks": tasks})
    except Exception as e:
        return jsonify({"ok": False, "err": str(e)}), 500

@app.post("/api/search-scores")
def search_scores_api():
    try:
        user = session.get("user")
        if not user: return jsonify({"success": False, "error": "未登入"}), 401

        target_uid = user.get("target_uid")
        data = request.get_json() or {}
        req_uid = data.get("uid", "").strip()
        req_task_id = data.get("task_id", "").strip()

        if req_task_id: req_task_id = normalize_task_id(req_task_id)
        search_uid = target_uid if target_uid else req_uid

        all_rows_raw = []
        tasks_to_search = [req_task_id] if req_task_id and req_task_id in TASK_MAP else TASK_MAP.keys()

        for tid in tasks_to_search:
            table_name = TASK_MAP[tid]
            sql = f"""
                SELECT d.uid, 
                       CASE 
                         WHEN u.name IS NOT NULL AND CHAR_LENGTH(u.name) > 0 THEN CONCAT(u.name, ' (', d.uid, ')')
                         ELSE d.uid 
                       END as name,
                       '{tid}' as task_id, 
                       d.score, d.result_img_path, d.test_date, d.time
                FROM `{table_name}` AS d
                JOIN user_list AS u ON u.uid = d.uid
                WHERE 1=1
            """
            params = []
            if search_uid:
                sql += " AND d.uid = %s"
                params.append(search_uid)

            rows = db_exec(sql, tuple(params), fetch="all") or []
            all_rows_raw.extend(rows)

        rows = _rows_date_to_str(all_rows_raw)
        
        task_names = {}
        try:
            t_rows = db_exec("SELECT task_id, task_name FROM task_list", fetch="all")
            for t in t_rows: task_names[t['task_id']] = t['task_name']
        except: pass

        for r in rows:
            uid = r.get("uid") or ""
            tid = r.get("task_id") or ""
            td  = r.get("test_date") or ""
            tm  = r.get("time") or "00:00:00"
            r["task_name"] = task_names.get(tid, tid)
            r["row_key"] = make_row_key(uid, tid, td, tm)
            
            base_filename = f"kid/{uid}/{tid}"
            candidates = []
            db_path = r.get('result_img_path')
            if db_path: candidates.append(db_path)
            if tid in MULTI_ANGLE_TASKS:
                candidates.append(f"{base_filename}-side.jpg")
                candidates.append(f"{base_filename}-top.jpg")
            candidates.append(f"{base_filename}.jpg")
            candidates.append(f"{base_filename}_result.jpg")
            
            real_path = None
            for p in candidates:
                if (ROOT / p).exists():
                    real_path = p
                    break
            
            if real_path:
                r['result_img_path'] = real_path
                r['compare_url'] = f"/view-compare?uid={uid}&task_id={tid}&img=/{real_path}"
            else:
                r['result_img_path'] = None
                r['compare_url'] = None

        rows.sort(key=lambda r: (r.get("test_date") or "", r.get("time") or ""), reverse=True)
        return jsonify({"success": True, "data": rows})

    except Exception as e:
        print(f"[Search API Error] {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/scores")
def list_scores():
    return search_scores_api()

# === Level 2 & 3: 新增/修改使用者個資 ===
@app.post("/api/user/upsert")
def upsert_user_only():
    try:
        user = session.get("user") or {}
        # Level 2 (一般管理) 和 Level 3 (超級管理) 都可以
        if int(user.get("level") or 0) < 2: return jsonify({"ok": False, "msg": "權限不足"}), 403
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()
        name = (data.get("name") or "").strip()
        birthday = (data.get("birthday") or "").strip()
        if not uid: return jsonify({"ok": False, "msg": "UID 為必填"}), 400
        db_exec("INSERT INTO user_list (uid, name, birthday) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE name=VALUES(name), birthday=VALUES(birthday)", (uid, name, birthday if birthday else None))
        return jsonify({"ok": True, "msg": "儲存成功"})
    except Exception as e: return jsonify({"ok": False, "msg": str(e)}), 500

# === Level 3 Only: 新增/修改關卡分數 ===
@app.post("/scores/upsert")
def upsert_score():
    try:
        user = session.get("user") or {}
        # 嚴格限制：只有 Level 3 可以
        if int(user.get("level") or 0) < 3: 
            return jsonify({"ok": False, "msg": "權限不足：只有超級管理者可修改分數"}), 403
        
        data = request.get_json() or {}
        uid = (data.get("uid") or "").strip()
        task_id = (data.get("task_id") or "").strip()
        score = int(data.get("score") or 0)
        test_date_str = (data.get("test_date") or "").strip()
        row_key_old = (data.get("row_key_old") or "").strip()

        if not uid or not task_id: return jsonify({"ok": False, "msg": "uid / task_id 不可為空"}), 400
        task_id = normalize_task_id(task_id)

        if test_date_str:
            try: test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()
            except: return jsonify({"ok": False, "msg": "日期格式錯誤"}), 400
        else: test_date = date.today()
        test_date_str = test_date.isoformat()
        
        current_time = datetime.now().strftime("%H:%M:%S")
        ensure_user(uid)
        ensure_task(task_id)

        if row_key_old:
            try:
                o_uid, o_tid, o_date, o_time = row_key_old.split("|", 3)
                db_exec("DELETE FROM score_list WHERE uid=%s AND task_id=%s AND test_date=%s AND time=%s", (o_uid, o_tid, o_date, o_time))
                o_tbl = TASK_MAP.get(o_tid)
                if o_tbl: db_exec(f"DELETE FROM `{o_tbl}` WHERE uid=%s AND test_date=%s AND time=%s", (o_uid, o_date, o_time))
            except: pass

        db_exec(
            "INSERT INTO score_list(uid, task_id, test_date, time) VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE test_date=VALUES(test_date), time=VALUES(time)",
            (uid, task_id, test_date, current_time)
        )
        
        table_name = task_id_to_table(task_id)
        db_exec(
            f"INSERT INTO `{table_name}`(uid, test_date, time, score) VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE score=VALUES(score), time=VALUES(time)",
            (uid, test_date, current_time, score)
        )
        return jsonify({"ok": True})
    except Exception as e: return jsonify({"ok": False, "msg": str(e)}), 500

# === Level 3 Only: 刪除關卡分數 ===
@app.delete("/scores")
def delete_score():
    try:
        user = session.get("user") or {}
        # 嚴格限制：只有 Level 3 可以
        if int(user.get("level") or 0) < 3: 
            return jsonify({"ok": False, "msg": "權限不足：只有超級管理者可刪除分數"}), 403
            
        row_key = (request.args.get("row_key") or "").strip()
        if not row_key: return jsonify({"ok": False, "msg": "缺少 row_key"}), 400
        try: uid, task_id, test_date, time_str = row_key.split("|", 3)
        except: return jsonify({"ok": False, "msg": "格式錯誤"}), 400
        db_exec("DELETE FROM score_list WHERE uid=%s AND task_id=%s AND test_date=%s AND time=%s", (uid, task_id, test_date, time_str))
        table = TASK_MAP.get(task_id)
        if table: db_exec(f"DELETE FROM `{table}` WHERE uid=%s AND test_date=%s AND time=%s", (uid, test_date, time_str))
        return jsonify({"ok": True})
    except Exception as e: return jsonify({"ok": False, "msg": str(e)}), 500

if __name__ == "__main__":
    try:
        print(f"啟動管理後台: http://{HOST}:{PORT}")
        threading.Timer(0.5, _open_browser).start()
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    except Exception as e: print(f"Server Error: {e}")