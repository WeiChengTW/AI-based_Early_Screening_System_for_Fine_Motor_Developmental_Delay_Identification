# run_admin.py
# -*- coding: utf-8 -*-
from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify, session
import threading
from datetime import datetime, date
import logging, uuid, os, secrets
from flask_cors import CORS
import traceback
from typing import Optional
from werkzeug.exceptions import HTTPException
import time
import pymysql

print("====== CURRENT ADMIN SERVER IS RUNNING (PORT 8001) ======")

DB = dict(
    host="100.117.109.112",
    port=3306,
    user="yplab",
    password="brain0918",
    database="tested",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

def db_exec(sql, params=None, fetch="none"):
    conn = pymysql.connect(**DB)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if fetch == "one": return cur.fetchone()
            if fetch == "all": return cur.fetchall()
            return None
    except Exception as e:
        write_to_console(f"[DB] PyMySQL 執行失敗: {sql}\nError: {e}", "ERROR")
        raise 
    finally:
        conn.close()

TASK_MAP = {
    "Ch1-t1": "string_blocks", "Ch1-t2": "pyramid", "Ch1-t3": "stair", "Ch1-t4": "build_wall",
    "Ch2-t1": "draw_circle", "Ch2-t2": "draw_square", "Ch2-t3": "draw_cross", "Ch2-t4": "draw_line",
    "Ch2-t5": "color", "Ch2-t6": "connect_dots",
    "Ch3-t1": "cut_circle", "Ch3-t2": "cut_square",
    "Ch4-t1": "one_fold", "Ch4-t2": "two_fold",
    "Ch5-t1": "collect_raisins",
}

def task_id_to_table(task_id: str) -> str:
    if task_id in TASK_MAP: return TASK_MAP[task_id]
    raise ValueError(f"未知的 task_id: {task_id}")

# 1. 修正 ensure_user 確保包含生日
def ensure_user(uid: str, name: Optional[str] = None, birthday: Optional[str] = None):
    db_exec(
        "INSERT INTO user_list(uid, name, birthday) VALUES (%s,%s,%s) "
        "ON DUPLICATE KEY UPDATE name=COALESCE(VALUES(name),name), birthday=COALESCE(VALUES(birthday),birthday)",
        (uid, name, birthday),
    )

def ensure_task(task_id: str):
    if task_id not in TASK_MAP: raise ValueError(f"未知的 task_id：{task_id}")
    task_name = TASK_MAP[task_id]
    db_exec("INSERT INTO task_list(task_id, task_name) VALUES (%s,%s) ON DUPLICATE KEY UPDATE task_name=VALUES(task_name)", (task_id, task_name))

def make_row_key(uid, task_id, test_date_str: str): return f"{uid}|{task_id}|{test_date_str}"

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

PORT = 8001
HOST = "127.0.0.1"
ROOT = Path(__file__).parent.resolve()

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app, supports_credentials=True)

def write_to_console(message, level="INFO"):
    console_path = ROOT / "console.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(console_path, "a", encoding="utf-8") as f: f.write(f"{ts} - {level} - {message}\n")
    except: pass

@app.errorhandler(Exception)
def _handle_err(e):
    if isinstance(e, HTTPException): return jsonify({"success": False, "error": str(e)}), e.code
    write_to_console(f"[ERR] {request.method} {request.path}\n{traceback.format_exc()}", "ERROR")
    return jsonify({"success": False, "error": str(e)}), 500

# 靜態檔案路由
@app.route("/admin")
@app.route("/admin.html")
def admin_shortcut(): return send_from_directory(ROOT / "html", "admin.html")
@app.route("/html/<path:filename>")
def html_files(filename): return send_from_directory(ROOT / "html", filename)
@app.route("/css/<path:filename>")
def css_files(filename): return send_from_directory(ROOT / "css", filename)
@app.route("/js/<path:filename>")
def js_files(filename): return send_from_directory(ROOT / "js", filename)
@app.route("/images/<path:filename>")
def images_files(filename): return send_from_directory(ROOT / "images", filename)

# -------------------------
# 身份驗證 API
# -------------------------
@app.post("/api/auth/login")
def api_login():
    data = request.get_json() or {}
    account, password = (data.get("account") or "").strip(), (data.get("password") or "").strip()
    if not account or not password: return jsonify({"ok": False, "msg": "請輸入帳號與密碼"}), 400
    row = db_exec("SELECT account, password, email, level FROM admin_users WHERE account=%s", (account,), fetch="one")
    if (not row) or row["password"] != password: return jsonify({"ok": False, "msg": "帳號或密碼錯誤"}), 401
    session["user"] = {"account": row["account"], "level": int(row.get("level") or 0), "name": row.get("email") or row["account"]}
    return jsonify({"ok": True, "user": session["user"]})

@app.get("/api/auth/whoami")
def api_whoami():
    user = session.get("user")
    return jsonify({"ok": True, "logged_in": True, "user": user}) if user else jsonify({"ok": True, "logged_in": False})

@app.post("/api/auth/logout")
def api_logout():
    session.pop("user", None)
    return jsonify({"ok": True})

# 🆕 新增：讓家長 (Level 1) 修改自己的帳號或密碼
@app.post("/api/auth/update_profile")
def api_update_profile():
    user = session.get("user")
    if not user: return jsonify({"ok": False, "msg": "未登入"}), 401
    
    data = request.get_json() or {}
    new_acc = data.get("new_account", "").strip()
    new_pwd = data.get("new_password", "").strip()
    old_acc = user["account"]

    if not new_acc or not new_pwd: return jsonify({"ok": False, "msg": "內容不可為空"}), 400

    try:
        # 更新 admin_users 列表
        db_exec("UPDATE admin_users SET account=%s, password=%s WHERE account=%s", (new_acc, new_pwd, old_acc))
        session.pop("user", None) # 修改完成後強制登出，要求重新登入
        return jsonify({"ok": True, "msg": "修改成功，請使用新憑證重新登入"})
    except Exception as e: return jsonify({"ok": False, "msg": str(e)}), 500

# -------------------------
# 資料操作 API
# -------------------------
@app.get("/scores")
def list_scores():
    try:
        user = session.get("user")
        if not user: return jsonify({"success": False, "error": "尚未登入"}), 401
        level, account = int(user.get("level") or 0), user.get("account")
        all_rows_raw = []
        for task_id, table_name in TASK_MAP.items():
            sql = f"""
                SELECT s.uid, u.name, s.task_id, t.task_name, d.score, d.result_img_path, s.test_date
                FROM score_list AS s
                JOIN user_list AS u ON u.uid = s.uid
                JOIN task_list AS t ON t.task_id = s.task_id
                LEFT JOIN `{table_name}` AS d ON d.uid = s.uid AND d.test_date = s.test_date
                WHERE s.task_id = %s
            """
            params = [task_id]
            if level == 1: # 🔐 家長過濾：帳號與 UID 綁定
                sql += " AND s.uid = %s"
                params.append(account)
            rows = db_exec(sql, tuple(params), fetch="all") or []
            all_rows_raw.extend(rows)
        
        def _date_to_str(rows):
            for r in rows or []:
                r = dict(r)
                td = r.get("test_date")
                if isinstance(td, (date, datetime)): r["test_date"] = td.isoformat()
                yield r
        
        rows = list(_date_to_str(all_rows_raw))
        for r in rows: r["row_key"] = make_row_key(r.get("uid") or "", r.get("task_id") or "", r.get("test_date") or "")
        rows.sort(key=lambda r: (r.get("test_date") or "", r.get("uid") or "", r.get("task_id") or ""), reverse=True)
        return jsonify(rows)
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.get("/users")
def list_users():
    try:
        user = session.get("user")
        if not user: return jsonify({"ok": False, "msg": "未登入"}), 401
        level, account = int(user.get("level") or 0), user.get("account")
        if level == 1:
            rows = db_exec("SELECT uid FROM user_list WHERE uid = %s ORDER BY uid", (account,), fetch="all")
        else:
            rows = db_exec("SELECT uid FROM user_list ORDER BY uid", fetch="all")
        return jsonify({"ok": True, "users": [r["uid"] for r in (rows or [])]})
    except Exception as e: return jsonify({"ok": False, "err": str(e)}), 500

# 2. 修改 api_add_user：連動建立家長帳號，帳號=UID, 密碼=生日
@app.post("/api/user/add")
def api_add_user():
    try:
        user_level = int(session.get("user", {}).get("level", 0))
        if user_level < 2: return jsonify({"ok": False, "msg": "權限不足"}), 403
        
        data = request.get_json() or {}
        uid = data.get("uid", "").strip()
        name = data.get("name", "").strip()
        birthday = data.get("birthday", "").strip()
        
        if not uid or not birthday: 
            return jsonify({"ok": False, "msg": "UID 與生日不可為空"}), 400

        # A. 寫入受測者基本資料
        ensure_user(uid, name, birthday)
        
        # B. 同步建立 Level 1 家長登入權限
        # 帳號預設為 uid, 密碼預設為 birthday
        db_exec(
            "INSERT INTO admin_users (account, password, email, level) VALUES (%s, %s, %s, 1) "
            "ON DUPLICATE KEY UPDATE password=COALESCE(password, VALUES(password))",
            (uid, birthday, f"{name}@parent.com")
        )
        return jsonify({"ok": True, "msg": "受測者與家長帳號已同步建立成功！"})
    except Exception as e: return jsonify({"ok": False, "msg": str(e)}), 500

# 🔐 Level 3 專用：手動修改紀錄
@app.post("/scores/upsert")
def upsert_score():
    try:
        user_level = int(session.get("user", {}).get("level", 0))
        if user_level < 3: 
            return jsonify({"ok": False, "msg": "只有主管(等級3)可以手動新增/修改分數"}), 403
            
        data = request.get_json() or {}
        uid = data.get("uid", "").strip()
        task_id = data.get("task_id", "").strip()
        score = int(data.get("score", 0))
        test_date_str = data.get("test_date", "").strip()
        
        if not uid or not task_id: return jsonify({"ok": False, "msg": "uid/task_id 不可為空"}), 400

        # 轉換日期
        test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date() if test_date_str else date.today()

        ensure_user(uid) # 確保 user_list 有這名小朋友
        ensure_task(task_id)

        # 1. 寫入總表：若已存在同人、同天、同關卡的紀錄，則不做任何事 (保持連結)
        db_exec(
            "INSERT INTO score_list (uid, task_id, test_date) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE test_date = VALUES(test_date)", 
            (uid, task_id, test_date)
        )

        # 2. 寫入任務表：若已存在同人、同天的紀錄，則直接更新分數 (達成「取最新筆」)
        table = task_id_to_table(task_id)
        db_exec(
            f"INSERT INTO `{table}` (uid, test_date, score) VALUES (%s, %s, %s) "
            f"ON DUPLICATE KEY UPDATE score = VALUES(score)", 
            (uid, test_date, score)
        )
        
        return jsonify({"ok": True, "msg": "紀錄已更新"})
    except Exception as e: return jsonify({"ok": False, "msg": str(e)}), 500

@app.delete("/scores")
def delete_score():
    try:
        if int(session.get("user", {}).get("level", 0)) < 3: return jsonify({"ok": False, "msg": "權限不足"}), 403
        row_key = request.args.get("row_key")
        if not row_key: return jsonify({"ok": False, "msg": "遺失 row_key"}), 400
        uid, task_id, test_date = row_key.split("|")
        table = task_id_to_table(task_id)
        db_exec(f"DELETE FROM `{table}` WHERE uid=%s AND test_date=%s", (uid, test_date))
        db_exec("DELETE FROM score_list WHERE uid=%s AND task_id=%s AND test_date=%s", (uid, task_id, test_date))
        return jsonify({"ok": True})
    except Exception as e: return jsonify({"ok": False, "msg": str(e)}), 500


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
