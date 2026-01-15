import os
import hashlib
import sqlite3
from urllib.parse import quote
from contextlib import contextmanager
from datetime import date

import psycopg
from psycopg.rows import tuple_row

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer, BadSignature
from dotenv import load_dotenv

load_dotenv()

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

# 本機預設 sqlite
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./local.db"

IS_SQLITE = DATABASE_URL.startswith("sqlite")
PH = "?" if IS_SQLITE else "%s"

# Render / 多數雲端 Postgres 常需要 SSL
if (not IS_SQLITE) and ("sslmode=" not in DATABASE_URL):
    DATABASE_URL += ("&" if "?" in DATABASE_URL else "?") + "sslmode=require"


# ======================
# DB helpers
# ======================
def get_conn():
    if IS_SQLITE:
        conn = sqlite3.connect("local.db")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    return psycopg.connect(DATABASE_URL, row_factory=tuple_row)


@contextmanager
def get_cursor(conn):
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def _sqlite_has_column(cur, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r["name"] for r in cur.fetchall()]
    return col in cols


def init_db():
    with get_conn() as conn:
        with get_cursor(conn) as cur:

            if IS_SQLITE:
                # 1 users
                cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                );
                """)

                # 2 records（✅加 face_value）
                cur.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_date DATE NOT NULL,
                    name TEXT NOT NULL,
                    face_value INTEGER NOT NULL DEFAULT 0,
                    total_amount INTEGER NOT NULL,
                    periods INTEGER NOT NULL,
                    amount INTEGER NOT NULL,
                    interval_days INTEGER NOT NULL,
                    paid_count INTEGER NOT NULL DEFAULT 0,
                    last_paid_day INTEGER NOT NULL DEFAULT 0,
                    user_id INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """)

                # 3 payments（一定最後）
                cur.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paid_at DATE NOT NULL,
                    amount INTEGER NOT NULL,
                    record_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES records(id) ON DELETE CASCADE,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """)

                # ✅ 自動補 records.face_value（舊資料庫才需要）
                cur.execute("PRAGMA table_info(records);")
                cols = [row[1] for row in cur.fetchall()]  # row[1] = 欄位名
                if "face_value" not in cols:
                    cur.execute("ALTER TABLE records ADD COLUMN face_value INTEGER NOT NULL DEFAULT 0;")

            else:
                # users
                cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                );
                """)

                # records（✅加 face_value）
                cur.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id SERIAL PRIMARY KEY,
                    created_date DATE NOT NULL,
                    name TEXT NOT NULL,
                    face_value INTEGER NOT NULL DEFAULT 0,
                    total_amount INTEGER NOT NULL,
                    periods INTEGER NOT NULL,
                    amount INTEGER NOT NULL,
                    interval_days INTEGER NOT NULL,
                    paid_count INTEGER NOT NULL DEFAULT 0,
                    last_paid_day INTEGER NOT NULL DEFAULT 0,
                    user_id INTEGER NOT NULL
                        REFERENCES users(id)
                        ON DELETE CASCADE
                );
                """)

                # payments
                cur.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id SERIAL PRIMARY KEY,
                    paid_at DATE NOT NULL,
                    amount INTEGER NOT NULL,
                    record_id INTEGER NOT NULL
                        REFERENCES records(id)
                        ON DELETE CASCADE,
                    user_id INTEGER NOT NULL
                        REFERENCES users(id)
                        ON DELETE CASCADE
                );
                """)

        conn.commit()


# ======================
# Utils
# ======================
def today_str() -> str:
    return date.today().isoformat()


def hash_password(password: str) -> str:
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
        return data
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
    return (date.today() - created_date_obj).days + 1


def calc_next_due_day(last_paid_day: int, interval_days: int) -> int:
    if last_paid_day <= 0:
        return interval_days
    return last_paid_day + interval_days


def row_to_view(row):
    rid = row["id"] if isinstance(row, sqlite3.Row) else row[0]
    created_date = row["created_date"] if isinstance(row, sqlite3.Row) else row[1]
    name = row["name"] if isinstance(row, sqlite3.Row) else row[2]
    face_value = row["face_value"] if isinstance(row, sqlite3.Row) else row[3]
    total_amount = row["total_amount"] if isinstance(row, sqlite3.Row) else row[4]
    periods = row["periods"] if isinstance(row, sqlite3.Row) else row[5]
    amount = row["amount"] if isinstance(row, sqlite3.Row) else row[6]
    interval_days = row["interval_days"] if isinstance(row, sqlite3.Row) else row[7]
    paid_count = row["paid_count"] if isinstance(row, sqlite3.Row) else row[8]
    last_paid_day = row["last_paid_day"] if isinstance(row, sqlite3.Row) else row[9]
    user_id = row["user_id"] if isinstance(row, sqlite3.Row) else row[10]

    if isinstance(created_date, str):
        y, m, d = created_date.split("-")
        created_date_obj = date(int(y), int(m), int(d))
    else:
        created_date_obj = created_date

    current_day = calc_current_day(created_date_obj)
    next_due_day = calc_next_due_day(int(last_paid_day), int(interval_days))
    days_left = next_due_day - current_day

    finished = int(paid_count) >= int(periods)
    is_due_today = (days_left == 0) and (not finished)
    is_overdue = (days_left < 0) and (not finished)

    return {
        "id": int(rid),
        "created_date": created_date_obj.isoformat(),
        "name": name,
        "face_value": int(face_value),
        "total_amount": int(total_amount),
        "periods": int(periods),
        "amount": int(amount),
        "interval_days": int(interval_days),
        "paid_count": int(paid_count),
        "last_paid_day": int(last_paid_day),
        "current_day": int(current_day),
        "next_due_day": int(next_due_day),
        "days_left": int(days_left),
        "finished": finished,
        "is_due_today": is_due_today,
        "is_overdue": is_overdue,
        "user_id": int(user_id),
    }


def get_all_records_for_user(user_id: int):
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"""
                SELECT id, created_date, name, face_value, total_amount, periods, amount,
                       interval_days, paid_count, last_paid_day, user_id
                FROM records
                WHERE user_id = {PH}
                ORDER BY id DESC
            """, (user_id,))
            rows = cur.fetchall()
            return [row_to_view(r) for r in rows]


def get_record_for_user(rid: int, user_id: int):
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"""
                SELECT id, created_date, name, face_value, total_amount, periods, amount,
                       interval_days, paid_count, last_paid_day, user_id
                FROM records
                WHERE id = {PH} AND user_id = {PH}
            """, (rid, user_id))
            row = cur.fetchone()
            return row_to_view(row) if row else None


def get_today_total_for_user(user_id: int) -> int:
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT COALESCE(SUM(amount), 0) FROM payments WHERE user_id = {PH} AND paid_at = {PH}",
                (user_id, today_str())
            )
            v = cur.fetchone()
            if isinstance(v, sqlite3.Row):
                return int(v[0])
            return int(v[0]) if v and v[0] is not None else 0


def get_history_grouped(user_id: int):
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"""
                SELECT id, paid_at, record_id, record_name, amount
                FROM payments
                WHERE user_id = {PH}
                ORDER BY record_id ASC, paid_at ASC, id ASC
            """, (user_id,))
            rows = cur.fetchall()

    # 計算每個 record 的第幾期
    seq = {}  # record_id -> count
    enriched = []

    for r in rows:
        pid = r["id"] if isinstance(r, sqlite3.Row) else r[0]
        paid_at = r["paid_at"] if isinstance(r, sqlite3.Row) else r[1]
        record_id = r["record_id"] if isinstance(r, sqlite3.Row) else r[2]
        record_name = r["record_name"] if isinstance(r, sqlite3.Row) else r[3]
        amount = r["amount"] if isinstance(r, sqlite3.Row) else r[4]

        seq[record_id] = seq.get(record_id, 0) + 1
        period_no = seq[record_id]

        paid_at_str = paid_at if isinstance(paid_at, str) else paid_at.isoformat()

        enriched.append({
            "id": int(pid),
            "date": paid_at_str,
            "name": record_name,
            "amount": int(amount),
            "period_no": int(period_no),
        })

    # 依日期分組（日期要倒序顯示）
    by_date = {}
    for it in enriched:
        d = it["date"]
        by_date.setdefault(d, {"date": d, "total": 0, "payments": []})
        by_date[d]["total"] += it["amount"]
        by_date[d]["payments"].append({
            "id": it["id"],
            "name": it["name"],
            "amount": it["amount"],
            "period_no": it["period_no"],
        })

    return sorted(by_date.values(), key=lambda x: x["date"], reverse=True)



# ======================
# Routes
# ======================
@app.get("/register")
def register_page():
    return RedirectResponse(url="/", status_code=303)


@app.get("/login")
def login_page():
    return RedirectResponse(url="/", status_code=303)


@app.get("/")
def home(request: Request):
    init_db()
    user = get_current_user(request)
    paid_msg = request.query_params.get("paid_msg", "")
    paid_id = request.query_params.get("paid_id")
    paid_id = int(paid_id) if paid_id and paid_id.isdigit() else None

    if not user:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "user": None, "paid_msg": paid_msg}
        )

    # ✅ 取出該使用者所有分期資料
    rows = get_all_records_for_user(user["user_id"])

    # ✅ 通知區：今日到期 / 逾期
    today_due_records = [r for r in rows if r["is_due_today"]]
    overdue_records = [r for r in rows if r["is_overdue"]]
    due_today_count = len(today_due_records)

    # ✅ 今日總收
    today_total = get_today_total_for_user(user["user_id"])

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "paid_msg": paid_msg,
            "paid_id": paid_id,
            "rows": rows,
            "today_due_records": today_due_records,
            "overdue_records": overdue_records,
            "due_today_count": due_today_count,
            "today_total": today_total,
        }
    )



@app.get("/add")
def add_page(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)

    return templates.TemplateResponse(
        "add.html",
        {"request": request, "user": user}
    )


@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    init_db()
    pw_hash = hash_password(password)

    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                if IS_SQLITE:
                    cur.execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, pw_hash)
                    )
                    user_id = cur.lastrowid
                else:
                    cur.execute(
                        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
                        (username, pw_hash)
                    )
                    user_id = cur.fetchone()[0]
            conn.commit()
    except Exception:
        return RedirectResponse(url="/?paid_msg=" + quote("帳號已存在或註冊失敗"), status_code=303)

    resp = RedirectResponse(url="/?paid_msg=" + quote("註冊成功，已登入"), status_code=303)
    return set_session_cookie(resp, user_id=int(user_id), username=username)


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    init_db()

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"SELECT id, password_hash FROM users WHERE username = {PH}", (username,))
            row = cur.fetchone()

    if not row:
        return RedirectResponse(url="/?paid_msg=" + quote("帳號不存在"), status_code=303)

    user_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
    pw_hash = row["password_hash"] if isinstance(row, sqlite3.Row) else row[1]

    if not verify_password(password, pw_hash):
        return RedirectResponse(url="/?paid_msg=" + quote("密碼錯誤"), status_code=303)

    resp = RedirectResponse(url="/?paid_msg=" + quote("登入成功"), status_code=303)
    return set_session_cookie(resp, user_id=int(user_id), username=username)


@app.post("/logout")
def logout():
    resp = RedirectResponse(url="/?paid_msg=" + quote("已登出"), status_code=303)
    return clear_session_cookie(resp)


@app.post("/add")
def add_record(
    request: Request,
    created_date: str = Form(...),
    name: str = Form(...),
    face_value: int = Form(0),
    total_amount: int = Form(...),
    amount: int = Form(...),
    periods: int = Form(...),
    interval_days: int = Form(...),
):
    user = require_login(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)

    init_db()
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"""
            INSERT INTO records
            (created_date, name, face_value, total_amount, periods, amount, interval_days, paid_count, last_paid_day, user_id)
            VALUES ({PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, 0, 0, {PH})
            """, (
                created_date, name, face_value, total_amount, periods, amount, interval_days, user["user_id"]
            ))
        conn.commit()

    return RedirectResponse(url="/?paid_msg=" + quote("新增完成"), status_code=303)

@app.get("/history")
def history(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)

    init_db()
    groups = get_history_grouped(user["user_id"])

    return templates.TemplateResponse(
        "history.html",
        {"request": request, "user": user, "groups": groups}
    )


@app.post("/history/delete")
def delete_payment(request: Request, payment_id: int = Form(...)):
    init_db()
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"""
                DELETE FROM payments
                WHERE id = {PH} AND user_id = {PH}
            """, (payment_id, user["user_id"]))
        conn.commit()

    return RedirectResponse("/history", status_code=303)


@app.get("/add-page")
def add_page(request: Request):
    user = require_login(request)
    if not user:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        "add.html",
        {
            "request": request,
            "user": user
        }
    )


@app.get("/ping")
def ping():
    return {"status": "ok"}


from datetime import date

@app.post("/pay/{record_id}")
def pay_record(request: Request, record_id: int):
    init_db()
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # 先拿資料（確認是自己的）
            cur.execute(f"""
                SELECT id, created_date, name, face_value, total_amount, periods, amount,
                       interval_days, paid_count, last_paid_day, user_id
                FROM records
                WHERE id = {PH} AND user_id = {PH}
            """, (record_id, user["user_id"]))
            row = cur.fetchone()
            if not row:
                return RedirectResponse("/", status_code=303)

            r = row_to_view(row)

            # 已繳清就不再新增
            if r["paid_count"] >= r["periods"]:
                return RedirectResponse("/", status_code=303)

            # ✅ 新增一筆繳款紀錄到 payments
            cur.execute(f"""
                INSERT INTO payments (paid_at, amount, record_id, user_id, record_name)
                VALUES ({PH}, {PH}, {PH}, {PH}, {PH})
            """, (date.today().isoformat(), int(r["amount"]), int(r["id"]), int(user["user_id"]), r["name"]))

            # ✅ 更新 records：paid_count +1、last_paid_day 更新成今天是第幾天
            cur.execute(f"""
                UPDATE records
                SET paid_count = paid_count + 1,
                    last_paid_day = {PH}
                WHERE id = {PH} AND user_id = {PH}
            """, (r["current_day"], r["id"], user["user_id"]))

        conn.commit()

    return RedirectResponse("/", status_code=303)


@app.post("/delete/{record_id}")
def delete_record(request: Request, record_id: int):
    init_db()
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"DELETE FROM records WHERE id = {PH} AND user_id = {PH}", (record_id, user["user_id"]))
        conn.commit()

    return RedirectResponse("/", status_code=303)


@app.get("/edit/{record_id}")
def edit_page(request: Request, record_id: int):
    init_db()
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"""
                SELECT id, created_date, name, face_value, total_amount, periods, amount,
                       interval_days, paid_count, last_paid_day, user_id
                FROM records
                WHERE id = {PH} AND user_id = {PH}
            """, (record_id, user["user_id"]))
            row = cur.fetchone()
            if not row:
                return RedirectResponse("/", status_code=303)

    r = row_to_view(row)
    return templates.TemplateResponse("edit.html", {"request": request, "user": user, "r": r})


@app.post("/edit/{record_id}")
def edit_save(
    request: Request,
    record_id: int,
    created_date: str = Form(...),
    name: str = Form(...),
    total_amount: int = Form(...),
    periods: int = Form(...),
    amount: int = Form(...),
    interval_days: int = Form(...),
):
    init_db()
    user = require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"""
                UPDATE records
                SET created_date = {PH},
                    name = {PH},
                    total_amount = {PH},
                    periods = {PH},
                    amount = {PH},
                    interval_days = {PH}
                WHERE id = {PH} AND user_id = {PH}
            """, (created_date, name, total_amount, periods, amount, interval_days, record_id, user["user_id"]))
        conn.commit()

    return RedirectResponse("/", status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))