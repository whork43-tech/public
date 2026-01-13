import os
import hashlib
from urllib.parse import quote

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from itsdangerous import URLSafeSerializer, BadSignature

import psycopg
from psycopg.rows import tuple_row
from datetime import date, datetime


# ======================
# App
# ======================
app = FastAPI(title="分期記帳")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ======================
# Env
# ======================
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")
serializer = URLSafeSerializer(SECRET_KEY, salt="session")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    # 上線一定要有 DATABASE_URL
    raise RuntimeError("Missing DATABASE_URL env var")

# Render / 多數雲端 Postgres 常需要 SSL
if "sslmode=" not in DATABASE_URL:
    DATABASE_URL += ("&" if "?" in DATABASE_URL else "?") + "sslmode=require"


# ======================
# DB helpers (Postgres)
# ======================
def get_conn():
    # tuple_row: 讓 fetch 回來是 tuple，方便沿用你原本的 row_to_view 解包
    return psycopg.connect(DATABASE_URL, row_factory=tuple_row)


def init_db():
    """
    建表（不存在才建立）
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # users
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            );
            """)

            # records
            cur.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id SERIAL PRIMARY KEY,
                created_date DATE NOT NULL,
                name TEXT NOT NULL,
                total_amount INTEGER NOT NULL,
                periods INTEGER NOT NULL,
                amount INTEGER NOT NULL,
                interval_days INTEGER NOT NULL,
                paid_count INTEGER NOT NULL DEFAULT 0,
                last_paid_day INTEGER NOT NULL DEFAULT 0,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
            );
            """)

        conn.commit()


# ======================
# Utils
# ======================
def today_str() -> str:
    return date.today().isoformat()


def hash_password(password: str) -> str:
    # 密碼雜湊（簡化版）
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


# ======================
# Session / Auth
# ======================
def set_session_cookie(resp: RedirectResponse, user_id: int, username: str):
    token = serializer.dumps({"user_id": user_id, "username": username})
    resp.set_cookie("session", token, httponly=True, samesite="lax")
    return resp


def clear_session_cookie(resp: RedirectResponse):
    resp.delete_cookie("session")
    return resp


def get_current_user(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        data = serializer.loads(token)
        return data  # {"user_id":..., "username":...}
    except BadSignature:
        return None


def require_login(request: Request):
    user = get_current_user(request)
    if not user:
        return None
    if not isinstance(user.get("user_id"), int):
        return None
    return user


# ======================
# Business logic
# ======================
def calc_current_day(created_date_obj: date) -> int:
    # created_date 當第 1 天
    return (date.today() - created_date_obj).days + 1


def calc_next_due_day(last_paid_day: int, interval_days: int) -> int:
    if last_paid_day <= 0:
        return interval_days
    return last_paid_day + interval_days


def row_to_view(row):
    # SELECT 欄位順序要對應這裡
    (rid, created_date, name, total_amount, periods, amount,
     interval_days, paid_count, last_paid_day, user_id) = row

    current_day = calc_current_day(created_date)
    next_due_day = calc_next_due_day(last_paid_day, interval_days)
    days_left = next_due_day - current_day

    finished = paid_count >= periods
    is_due_today = (days_left == 0) and (not finished)
    is_overdue = (days_left < 0) and (not finished)

    return {
        "id": rid,
        "created_date": created_date.isoformat(),
        "name": name,
        "total_amount": total_amount,
        "periods": periods,
        "amount": amount,
        "interval_days": interval_days,
        "paid_count": paid_count,
        "last_paid_day": last_paid_day,
        "current_day": current_day,
        "next_due_day": next_due_day,
        "days_left": days_left,
        "finished": finished,
        "is_due_today": is_due_today,
        "is_overdue": is_overdue,
        "user_id": user_id,
    }


def get_all_records_for_user(user_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, created_date, name, total_amount, periods, amount,
                       interval_days, paid_count, last_paid_day, user_id
                FROM records
                WHERE user_id = %s
                ORDER BY id DESC
            """, (user_id,))
            rows = cur.fetchall()
    return [row_to_view(r) for r in rows]


def get_record_for_user(rid: int, user_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, created_date, name, total_amount, periods, amount,
                       interval_days, paid_count, last_paid_day, user_id
                FROM records
                WHERE id = %s AND user_id = %s
            """, (rid, user_id))
            row = cur.fetchone()
    return row_to_view(row) if row else None


# ======================
# Routes
# ======================
@app.get("/")
def home(request: Request):
    init_db()

    user = get_current_user(request)

    paid_msg = request.query_params.get("paid_msg", "")
    paid_id_str = request.query_params.get("paid_id", "")
    paid_id = int(paid_id_str) if paid_id_str.isdigit() else None

    if not user:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": None,
                "rows": [],
                "today_due_records": [],
                "overdue_records": [],
                "due_today_count": 0,
                "paid_msg": paid_msg,
                "paid_id": paid_id,
            }
        )

    records = get_all_records_for_user(user["user_id"])
    today_due_records = [r for r in records if r["is_due_today"]]
    overdue_records = [r for r in records if r["is_overdue"]]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "today": today_str(),
            "rows": records,
            "today_due_records": today_due_records,
            "overdue_records": overdue_records,
            "due_today_count": len(today_due_records),
            "paid_msg": paid_msg,
            "paid_id": paid_id,
        }
    )


@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    init_db()
    pw_hash = hash_password(password)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
                    (username, pw_hash)
                )
                user_id = cur.fetchone()[0]
            conn.commit()
    except Exception:
        return RedirectResponse(url="/?paid_msg=" + quote("帳號已存在或註冊失敗"), status_code=303)

    resp = RedirectResponse(url="/?paid_msg=" + quote("註冊成功，已登入"), status_code=303)
    return set_session_cookie(resp, user_id=user_id, username=username)


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    init_db()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            row = cur.fetchone()

    if not row:
        return RedirectResponse(url="/?paid_msg=" + quote("帳號不存在"), status_code=303)

    user_id, pw_hash = row[0], row[1]
    if not verify_password(password, pw_hash):
        return RedirectResponse(url="/?paid_msg=" + quote("密碼錯誤"), status_code=303)

    resp = RedirectResponse(url="/?paid_msg=" + quote("登入成功"), status_code=303)
    return set_session_cookie(resp, user_id=user_id, username=username)


@app.post("/logout")
def logout():
    resp = RedirectResponse(url="/?paid_msg=" + quote("已登出"), status_code=303)
    return clear_session_cookie(resp)


@app.post("/add")
def add_record(
    request: Request,
    created_date: str = Form(...),
    name: str = Form(...),
    total_amount: int = Form(...),
    periods: int = Form(...),
    amount: int = Form(...),
    interval_days: int = Form(...),
):
    user = require_login(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)

    init_db()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO records
                (created_date, name, total_amount, periods, amount, interval_days, paid_count, last_paid_day, user_id)
                VALUES (%s, %s, %s, %s, %s, %s, 0, 0, %s)
            """, (created_date, name, total_amount, periods, amount, interval_days, user["user_id"]))
        conn.commit()

    return RedirectResponse(url="/", status_code=303)


@app.post("/pay/{rid}")
def mark_paid(request: Request, rid: int):
    user = require_login(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)

    init_db()
    rec = get_record_for_user(rid, user["user_id"])
    if not rec:
        return RedirectResponse(url="/?paid_msg=" + quote("找不到資料"), status_code=303)

    if not (rec["is_due_today"] or rec["is_overdue"]):
        return RedirectResponse(url="/?paid_msg=" + quote("尚未到繳款日"), status_code=303)

    new_paid_count = rec["paid_count"] + 1
    new_last_paid_day = rec["next_due_day"]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE records
                SET paid_count = %s, last_paid_day = %s
                WHERE id = %s AND user_id = %s
            """, (new_paid_count, new_last_paid_day, rid, user["user_id"]))
        conn.commit()

    msg = quote(f"{rec['name']} 繳款完成")
    return RedirectResponse(url=f"/?paid_msg={msg}&paid_id={rid}", status_code=303)


@app.post("/delete/{rid}")
def delete_record(request: Request, rid: int):
    user = require_login(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)

    init_db()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM records WHERE id = %s AND user_id = %s", (rid, user["user_id"]))
        conn.commit()

    return RedirectResponse(url="/?paid_msg=" + quote("已刪除"), status_code=303)


# (可留著) 健康檢查用
@app.get("/ping")
def ping():
    return {"status": "ok"}