import os
from fastapi.responses import HTMLResponse
import hashlib
import sqlite3
from urllib.parse import quote
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import tuple_row
import traceback
from fastapi.responses import PlainTextResponse

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer, BadSignature
from dotenv import load_dotenv


def taipei_today_str() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat()


load_dotenv()

# ======================
# App
# ======================
app = FastAPI(title="分期記帳")


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    print("\n=== UNHANDLED ERROR ===")
    print(f"{request.method} {request.url}")
    print(tb)
    print("=== END ERROR ===\n")
    print(
        "DB:",
        "sqlite" if IS_SQLITE else "postgres",
        "DATABASE_URL startswith:",
        DATABASE_URL[:20],
    )

    return PlainTextResponse(tb, status_code=500)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ======================
# Env
# ======================
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")
serializer = URLSafeSerializer(SECRET_KEY, salt="session")

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "").strip()
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


def get_paid_sum_map(user_id: int) -> dict[int, int]:
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT record_id, COALESCE(SUM(amount), 0) AS s
                FROM payments
                WHERE user_id = {PH}
                GROUP BY record_id
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    out = {}
    for r in rows:
        record_id = r["record_id"] if isinstance(r, sqlite3.Row) else r[0]
        s = r["s"] if isinstance(r, sqlite3.Row) else r[1]

        # ✅ 防呆：跳過 record_id 為 NULL 的壞資料
        if record_id is None:
            continue

        out[int(record_id)] = int(s or 0)
    return out


@contextmanager
def get_cursor(conn):
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def _sqlite_has_column(cur, table: str, col: str) -> bool:
    # ✅ 防呆：只允許 SQLite 執行 PRAGMA
    if not IS_SQLITE:
        return True  # 回 True：即使被誤叫，也不會進入 ALTER ADD COLUMN
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r["name"] for r in cur.fetchall()]
    return col in cols


def init_db():
    with get_conn() as conn:
        with get_cursor(conn) as cur:

            # ========== USERS ==========
            if IS_SQLITE:
                cur.execute(
                    """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                );
                """
                )

                # ✅ SQLite 才能用 PRAGMA 檢查欄位
                if not _sqlite_has_column(cur, "users", "activated_at"):
                    cur.execute("ALTER TABLE users ADD COLUMN activated_at TEXT")

                if not _sqlite_has_column(cur, "users", "display_name"):
                    cur.execute("ALTER TABLE users ADD COLUMN display_name TEXT")

                if not _sqlite_has_column(cur, "users", "trial_until"):
                    cur.execute("ALTER TABLE users ADD COLUMN trial_until TEXT")

                if not _sqlite_has_column(cur, "users", "last_seen_at"):
                    cur.execute("ALTER TABLE users ADD COLUMN last_seen_at TEXT")

                if not _sqlite_has_column(cur, "users", "group_activated_at"):
                    cur.execute("ALTER TABLE users ADD COLUMN group_activated_at TEXT")

                if not _sqlite_has_column(cur, "users", "group_trial_started_at"):
                    cur.execute(
                        "ALTER TABLE users ADD COLUMN group_trial_started_at TEXT"
                    )

            else:
                cur.execute(
                    """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                );
                """
                )

                # ✅ Postgres 直接用 IF NOT EXISTS（不用 PRAGMA）
                cur.execute(
                    """
                ALTER TABLE users
                ADD COLUMN IF NOT EXISTS activated_at TEXT
                """
                )

                cur.execute(
                    """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS display_name TEXT
"""
                )

                cur.execute(
                    """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS trial_until TEXT
"""
                )

                cur.execute(
                    """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS last_seen_at TEXT
"""
                )

                cur.execute(
                    """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS group_activated_at TEXT
"""
                )

                cur.execute(
                    """
ALTER TABLE users
ADD COLUMN IF NOT EXISTS group_trial_started_at TEXT
"""
                )

                # ========== USERS：連結功能欄位 ==========
            if IS_SQLITE:
                if not _sqlite_has_column(cur, "users", "group_activated_at"):
                    cur.execute("ALTER TABLE users ADD COLUMN group_activated_at TEXT")
                if not _sqlite_has_column(cur, "users", "group_trial_started_at"):
                    cur.execute(
                        "ALTER TABLE users ADD COLUMN group_trial_started_at TEXT"
                    )
            else:
                cur.execute(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS group_activated_at TEXT"
                )
                cur.execute(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS group_trial_started_at TEXT"
                )

            # ========== ACCOUNT LINKS ==========
            if IS_SQLITE:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS account_links (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        master_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        child_user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                    """
                )
                # 子帳號只能被連結一次（避免被多個主帳號綁走）
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_links_child ON account_links(child_user_id)"
                )
                # 同一主帳號重複連同一子帳號也擋
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_links_pair ON account_links(master_user_id, child_user_id)"
                )
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS account_links (
                        id SERIAL PRIMARY KEY,
                        master_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        child_user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        created_at TEXT NOT NULL DEFAULT (now()::text),
                        UNIQUE(child_user_id),
                        UNIQUE(master_user_id, child_user_id)
                    );
                    """
                )

                # ========== RECORDS ==========
            if IS_SQLITE:
                cur.execute(
                    """
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
                        user_id INTEGER NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE
                    );
                    """
                )

                # SQLite：用 PRAGMA 檢查欄位再 ADD COLUMN（避免 IF NOT EXISTS 語法不支援）
                if not _sqlite_has_column(cur, "records", "use_face_value"):
                    cur.execute(
                        "ALTER TABLE records ADD COLUMN use_face_value BOOLEAN NOT NULL DEFAULT 1"
                    )
                if not _sqlite_has_column(cur, "records", "is_deleted"):
                    cur.execute(
                        "ALTER TABLE records ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT 0"
                    )
                if not _sqlite_has_column(cur, "records", "ticket_offset"):
                    cur.execute(
                        "ALTER TABLE records ADD COLUMN ticket_offset INTEGER NOT NULL DEFAULT 0"
                    )
                if not _sqlite_has_column(cur, "records", "expense_offset"):
                    cur.execute(
                        "ALTER TABLE records ADD COLUMN expense_offset INTEGER NOT NULL DEFAULT 0"
                    )

            else:
                cur.execute(
                    """
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
                    """
                )

                # Postgres：可用 IF NOT EXISTS
                cur.execute(
                    """
                    ALTER TABLE records
                    ADD COLUMN IF NOT EXISTS use_face_value BOOLEAN NOT NULL DEFAULT TRUE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE records
                    ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE records
                    ADD COLUMN IF NOT EXISTS ticket_offset INTEGER NOT NULL DEFAULT 0
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE records
                    ADD COLUMN IF NOT EXISTS expense_offset INTEGER NOT NULL DEFAULT 0
                    """
                )

            # ========== PAYMENTS ==========
            if IS_SQLITE:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS payments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        paid_at DATE NOT NULL,
                        amount INTEGER NOT NULL,
                        record_id INTEGER NOT NULL
                            REFERENCES records(id)
                            ON DELETE CASCADE,
                        record_name TEXT NOT NULL DEFAULT '',
                        user_id INTEGER NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE
                    );
                    """
                )

                # 舊資料庫可能沒有 record_name：補欄位
                if not _sqlite_has_column(cur, "payments", "record_name"):
                    cur.execute(
                        "ALTER TABLE payments ADD COLUMN record_name TEXT NOT NULL DEFAULT ''"
                    )

            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS payments (
                        id SERIAL PRIMARY KEY,
                        paid_at DATE NOT NULL,
                        amount INTEGER NOT NULL,
                        record_id INTEGER NOT NULL
                            REFERENCES records(id)
                            ON DELETE CASCADE,
                        record_name TEXT NOT NULL DEFAULT '',
                        user_id INTEGER NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE
                    );
                    """
                )

                # 舊資料庫可能沒有 record_name：補欄位
                cur.execute(
                    """
                    ALTER TABLE payments
                    ADD COLUMN IF NOT EXISTS record_name TEXT NOT NULL DEFAULT ''
                    """
                )

            # ========== EXPENSES ==========
            if IS_SQLITE:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS expenses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        spent_at DATE NOT NULL,
                        item TEXT NOT NULL,
                        amount INTEGER NOT NULL,
                        user_id INTEGER NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE
                    );
                    """
                )
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS expenses (
                        id SERIAL PRIMARY KEY,
                        spent_at DATE NOT NULL,
                        item TEXT NOT NULL,
                        amount INTEGER NOT NULL,
                        user_id INTEGER NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE
                    );
                    """
                )

            # ========== ACCOUNT LINKS (Master -> Child) ==========
            # 主帳號可以看到子帳號的彙總數字（不影響各帳號原本資料與功能）
            if IS_SQLITE:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS account_links (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        master_user_id INTEGER NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE,
                        child_user_id INTEGER NOT NULL
                            REFERENCES users(id)
                            ON DELETE CASCADE,
                        UNIQUE(master_user_id, child_user_id)
                    );
                    """
                )
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS account_links (
                        id SERIAL PRIMARY KEY,
                        master_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        child_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        UNIQUE(master_user_id, child_user_id)
                    );
                    """
                )

        conn.commit()


# ======================
# Utils
# ======================
def today_str() -> str:
    return taipei_today_str()


def touch_last_seen(user_id: int):
    today = today_str()

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                UPDATE users
                SET last_seen_at = {PH}
                WHERE id = {PH}
                  AND (last_seen_at IS NULL OR last_seen_at != {PH})
                """,
                (today, user_id, today),
            )
        conn.commit()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


# ======================
# Billing / Access control
# ======================


def _parse_iso_date(s: str | None):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def get_group_access_status(user_id: int):
    """
    連結功能狀態：
    - 第一次使用：自動啟動 7 天試用（寫入 group_trial_started_at）
    - 試用中：OK
    - 付費中：OK（group_activated_at >= today）
    - 到期：不給用
    回傳 (ok, mode, until, msg)
    mode: "activated" | "trial" | "expired"
    """
    init_db()
    today = date.fromisoformat(today_str())

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT group_activated_at, group_trial_started_at FROM users WHERE id = {PH}",
                (user_id,),
            )
            row = cur.fetchone()

    if not row:
        return (False, "expired", None, "帳號不存在")

    if isinstance(row, sqlite3.Row):
        activated_s = row["group_activated_at"]
        trial_s = row["group_trial_started_at"]
    else:
        activated_s = row[0]
        trial_s = row[1]

    activated = _parse_iso_date(activated_s)
    trial_start = _parse_iso_date(trial_s)

    # 1) 已付費開通
    if activated and today <= activated:
        return (True, "activated", activated, "")

    # 2) 試用已開始
    if trial_start:
        trial_until = trial_start + timedelta(days=7)
        if today <= trial_until:
            return (True, "trial", trial_until, "")
        return (
            False,
            "expired",
            None,
            "連結功能試用已到期，請聯絡 LINE：@826ynmlh 開通",
        )

    # 3) 第一次使用：啟動試用
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"UPDATE users SET group_trial_started_at = {PH} WHERE id = {PH}",
                (today.isoformat(), user_id),
            )
        conn.commit()

    return (True, "trial", today + timedelta(days=7), "")


def get_access_status(user_id: int):
    """
    回傳 (ok, msg, until, mode)
    mode: activated | trial | expired
    """
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT activated_at, trial_until FROM users WHERE id = {PH}",
                (user_id,),
            )
            row = cur.fetchone()

    if not row:
        return (False, "帳號不存在", None, "expired")

    if isinstance(row, sqlite3.Row):
        activated_at = row["activated_at"]
        trial_until = row["trial_until"]
    else:
        activated_at = row[0]
        trial_until = row[1]

    today = date.fromisoformat(today_str())
    act = _parse_iso_date(activated_at)
    tr = _parse_iso_date(trial_until)

    # ✅ 已開通（優先）
    if act and today <= act:
        return (True, "", act, "activated")

    # ✅ 試用期內（尚未開通）
    if (not act) and tr and today <= tr:
        return (True, "", tr, "trial")

    # ✅ 未開通或已到期
    return (
        False,
        "試用期結束.請與管理員聯絡LINE:@826ynmlh",
        None,
        "expired",
    )


# ======================
# Session / Auth
# ======================
def set_session_cookie(
    resp: RedirectResponse, user_id: int, username: str, display_name: str | None = None
):
    token = serializer.dumps(
        {"user_id": user_id, "username": username, "display_name": display_name or ""}
    )
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
    init_db()  # ✅ 確保欄位都存在
    user = get_current_user(request)
    if not user:
        return None
    if not isinstance(user.get("user_id"), int):
        return None
    touch_last_seen(user["user_id"])
    return user


def require_active(request: Request):
    user = require_login(request)
    if not user:
        return None

    ok, msg, _, _ = get_access_status(user["user_id"])
    if not ok:
        return RedirectResponse(url="/?paid_msg=" + quote(msg), status_code=303)

    return user


def require_admin(request: Request):
    user = require_login(request)
    if not user:
        return None
    # 用登入者的 username 判斷是否管理員
    if ADMIN_USERNAME and user.get("username") == ADMIN_USERNAME:
        return user
    return None


def get_child_user_ids(master_user_id: int) -> list[int]:
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT child_user_id FROM account_links WHERE master_user_id = {PH} ORDER BY child_user_id ASC",
                (master_user_id,),
            )
            rows = cur.fetchall()

    out: list[int] = []
    for r in rows:
        cid = r["child_user_id"] if isinstance(r, sqlite3.Row) else r[0]
        try:
            out.append(int(cid))
        except Exception:
            pass
    return out


def is_child_user(user_id: int) -> bool:
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT 1 FROM account_links WHERE child_user_id = {PH} LIMIT 1",
                (user_id,),
            )
            row = cur.fetchone()
    return bool(row)


def is_master_user(user_id: int) -> bool:
    # 沒有被別人連結成子帳號，就允許當主帳號（是否「付費/試用可用」再交給 group access 判斷）
    return not is_child_user(user_id)


def compute_user_face_totals(user_id: int) -> tuple[int, int]:
    """
    回傳 (總票面, 總票面餘)；只算未結清 (paid_count < periods)
    規則沿用你首頁原本的算法，不動既有功能
    """
    rows_raw = get_all_records_for_user(user_id)
    rows = [row_to_view(r) for r in rows_raw]
    paid_sum_map = get_paid_sum_map(user_id)
    for r in rows:
        r["paid_sum"] = paid_sum_map.get(r["id"], 0)

    total_face_value = sum(
        int((r.get("face_value") or 0))
        for r in rows
        if int(r.get("paid_count", 0)) < int(r.get("periods", 0))
    )

    total_face_value_left = sum(
        int((r.get("face_value") or 0))
        - int((r.get("ticket_offset") or 0))
        - int((r.get("paid_sum") or 0))
        for r in rows
        if int(r.get("paid_count", 0)) < int(r.get("periods", 0))
    )

    return int(total_face_value), int(total_face_value_left)


def compute_group_summary(
    master_user_id: int, year: int | None = None, month: int | None = None
) -> dict:
    """
    只計算『連結帳號』的指定月份淨利 + 帳號顯示名稱
    """
    child_ids = get_child_user_ids(master_user_id)

    if not child_ids:
        return {"month_total": 0, "per_user": []}

    placeholders = ",".join([PH] * len(child_ids))

    # 一次抓 username/display_name
    user_map = {}
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT id, username, display_name FROM users WHERE id IN ({placeholders})",
                tuple(child_ids),
            )
            rows = cur.fetchall()

    for r in rows:
        if isinstance(r, sqlite3.Row):
            uid = int(r["id"])
            user_map[uid] = {
                "username": r["username"],
                "display_name": r["display_name"] or "",
            }
        else:
            uid = int(r[0])
            user_map[uid] = {"username": r[1], "display_name": r[2] or ""}

    per_user = []
    total = 0

    for uid in child_ids:
        month_net = get_month_net_for_user(uid, year=year, month=month)
        total += month_net
        info = user_map.get(uid, {"username": "", "display_name": ""})

        per_user.append(
            {
                "user_id": uid,
                "username": info["username"],
                "display_name": info["display_name"],
                "month_net": int(month_net),
            }
        )

    # ✅ 排序：淨利高到低
    per_user.sort(key=lambda x: x["month_net"], reverse=True)

    return {"month_total": int(total), "per_user": per_user}


from datetime import date


def _month_start_end(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def get_month_net_for_user(
    user_id: int, year: int | None = None, month: int | None = None
) -> int:
    """
    計算單一帳號的『指定月份淨利』
    淨利 = 該月實收 - 該月開銷
    """
    today = date.fromisoformat(today_str())  # ✅ 跟全站一致（Asia/Taipei）
    y = int(year or today.year)
    m = int(month or today.month)

    month_start, month_end = _month_start_end(y, m)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # 該月實收
            cur.execute(
                f"""
                SELECT COALESCE(SUM(amount), 0)
                FROM payments
                WHERE user_id = {PH}
                  AND paid_at >= {PH}
                  AND paid_at < {PH}
                """,
                (user_id, month_start, month_end),
            )
            income = cur.fetchone()[0] or 0

            # 該月開銷
            cur.execute(
                f"""
                SELECT COALESCE(SUM(amount), 0)
                FROM expenses
                WHERE user_id = {PH}
                  AND spent_at >= {PH}
                  AND spent_at < {PH}
                """,
                (user_id, month_start, month_end),
            )
            expense = cur.fetchone()[0] or 0

    return int(income) - int(expense)


# ======================
# Business logic
# ======================
def calc_current_day(created_date_obj: date) -> int:
    today = date.fromisoformat(taipei_today_str())
    return (today - created_date_obj).days


def get_linked_accounts(master_user_id: int) -> list[dict]:
    ids = get_child_user_ids(master_user_id)
    if not ids:
        return []

    phs = ",".join([PH] * len(ids))
    out = []

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT id, username, display_name, activated_at, trial_until
                FROM users
                WHERE id IN ({phs})
                """,
                tuple(ids),
            )
            rows = cur.fetchall()

    today = date.fromisoformat(today_str())

    # 做 map，確保順序跟 ids 一樣
    tmp = {}
    for r in rows:
        if isinstance(r, sqlite3.Row):
            uid = int(r["id"])
            activated_at = _parse_iso_date(r["activated_at"])
            trial_until = _parse_iso_date(r["trial_until"])
            tmp[uid] = {
                "user_id": uid,
                "username": r["username"],
                "display_name": r["display_name"] or "",
                "main_active": bool(
                    (activated_at and today <= activated_at)
                    or (trial_until and today <= trial_until)
                ),
            }
        else:
            uid = int(r[0])
            activated_at = _parse_iso_date(r[3])
            trial_until = _parse_iso_date(r[4])
            tmp[uid] = {
                "user_id": uid,
                "username": r[1],
                "display_name": r[2] or "",
                "main_active": bool(
                    (activated_at and today <= activated_at)
                    or (trial_until and today <= trial_until)
                ),
            }

    for uid in ids:
        out.append(
            tmp.get(
                uid,
                {
                    "user_id": uid,
                    "username": "",
                    "display_name": "",
                    "main_active": False,
                },
            )
        )

    return out


def calc_next_due_day(last_paid_day: int, interval_days: int) -> int:
    if last_paid_day <= 0:
        return interval_days  # ✅ 建立日當天不算收款日
    return last_paid_day + interval_days


def row_to_view(row):
    rid = row["id"] if isinstance(row, sqlite3.Row) else row[0]
    created_date = row["created_date"] if isinstance(row, sqlite3.Row) else row[1]
    name = row["name"] if isinstance(row, sqlite3.Row) else row[2]
    face_value = row["face_value"] if isinstance(row, sqlite3.Row) else row[3]
    ticket_offset = row["ticket_offset"] if isinstance(row, sqlite3.Row) else row[11]
    expense_offset = row["expense_offset"] if isinstance(row, sqlite3.Row) else row[12]
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
        "ticket_offset": int(ticket_offset or 0),
        "expense_offset": int(expense_offset or 0),
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
            try:
                # ✅ 新版：有 is_deleted 就用
                cur.execute(
                    f"""
    SELECT id, created_date, name, face_value, total_amount, periods, amount,
           interval_days, paid_count, last_paid_day, user_id, ticket_offset, expense_offset
    FROM records
    WHERE user_id = {PH}
      AND is_deleted = FALSE
    ORDER BY id DESC
    """,
                    (user_id,),
                )

            except Exception as e:
                # 只針對沒有 is_deleted 欄位時降級
                if "is_deleted" not in str(e):
                    raise

                # ✅ 关键：Postgres 第一個 SQL 失敗後，必須 rollback 才能繼續下 SQL
                if not IS_SQLITE:
                    conn.rollback()

                # ✅ 舊版：沒有 is_deleted 就退回原本查詢（不影響既有功能）
                cur.execute(
                    f"""
    SELECT id, created_date, name, face_value, total_amount, periods, amount,
           interval_days, paid_count, last_paid_day, user_id, ticket_offset, expense_offset
    FROM records
    WHERE user_id = {PH}
    ORDER BY id DESC
    """,
                    (user_id,),
                )

            return cur.fetchall()


def get_record_for_user(rid: int, user_id: int):
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
    SELECT id, created_date, name, face_value, total_amount, periods, amount,
           interval_days, paid_count, last_paid_day, user_id, ticket_offset, expense_offset
    FROM records
    WHERE id = {PH} AND user_id = {PH}
""",
                (rid, user_id),
            )

            row = cur.fetchone()
            return row_to_view(row) if row else None


def get_today_total_for_user(user_id: int) -> int:
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT COALESCE(SUM(amount), 0) FROM payments WHERE user_id = {PH} AND paid_at = {PH}",
                (user_id, today_str()),
            )
            v = cur.fetchone()
            if isinstance(v, sqlite3.Row):
                return int(v[0])
            return int(v[0]) if v and v[0] is not None else 0


def get_today_expenses_for_user(user_id: int):
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT id, item, amount
                FROM expenses
                WHERE user_id = {PH} AND spent_at = {PH}
                ORDER BY id DESC
                """,
                (user_id, today_str()),
            )

            rows = cur.fetchall()

    out = []
    for r in rows:
        if isinstance(r, sqlite3.Row):
            out.append(
                {"id": int(r["id"]), "item": r["item"], "amount": int(r["amount"])}
            )
        else:
            out.append({"id": int(r[0]), "item": r[1], "amount": int(r[2])})
    return out


def get_today_expense_total_for_user(user_id: int) -> int:
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE user_id = {PH} AND spent_at = {PH}",
                (user_id, today_str()),
            )
            v = cur.fetchone()
            if isinstance(v, sqlite3.Row):
                return int(v[0])
            return int(v[0]) if v and v[0] is not None else 0


def get_expense_map_for_user(user_id: int):
    """
    回傳 dict:
    {
      "YYYY-MM-DD": {"date": "...", "total": 123, "expenses": [{"item": "...", "amount": 50}, ...]},
      ...
    }
    """
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT spent_at, item, amount
                FROM expenses
                WHERE user_id = {PH}
                ORDER BY spent_at DESC, id DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    mp = {}
    for r in rows:
        if isinstance(r, sqlite3.Row):
            spent_at = r["spent_at"]
            item = r["item"]
            amount = r["amount"]
        else:
            spent_at, item, amount = r[0], r[1], r[2]

        d = spent_at if isinstance(spent_at, str) else spent_at.isoformat()

        if d not in mp:
            mp[d] = {"date": d, "total": 0, "expenses": []}

        mp[d]["total"] += int(amount or 0)
        mp[d]["expenses"].append({"item": item, "amount": int(amount or 0)})

    return mp


def get_history_grouped(user_id: int):
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT p.id, p.paid_at, p.record_id, r.name, p.amount
                FROM payments p
                JOIN records r ON r.id = p.record_id
                WHERE p.user_id = {PH}
                ORDER BY p.record_id ASC, p.paid_at ASC, p.id ASC
            """,
                (user_id,),
            )
            rows = cur.fetchall()

    # 計算每個 record 的第幾期
    seq = {}
    enriched = []

    for r in rows:
        pid = r["id"] if isinstance(r, sqlite3.Row) else r[0]
        paid_at = r["paid_at"] if isinstance(r, sqlite3.Row) else r[1]
        record_id = r["record_id"] if isinstance(r, sqlite3.Row) else r[2]
        record_name = r["name"] if isinstance(r, sqlite3.Row) else r[3]
        amount = r["amount"] if isinstance(r, sqlite3.Row) else r[4]

        seq[record_id] = seq.get(record_id, 0) + 1
        period_no = seq[record_id]

        paid_at_str = paid_at if isinstance(paid_at, str) else paid_at.isoformat()

        enriched.append(
            {
                "id": int(pid),
                "date": paid_at_str,
                "name": record_name,
                "amount": int(amount),
                "period_no": int(period_no),
            }
        )

    # 依日期分組（日期要倒序顯示）
    by_date = {}
    for it in enriched:
        d = it["date"]
        by_date.setdefault(d, {"date": d, "total": 0, "payments": []})
        by_date[d]["total"] += it["amount"]
        by_date[d]["payments"].append(
            {
                "id": it["id"],
                "name": it["name"],
                "amount": it["amount"],
                "period_no": it["period_no"],
            }
        )

    return sorted(by_date.values(), key=lambda x: x["date"], reverse=True)


def get_expense_history_grouped(user_id: int):
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT spent_at, item, amount
                FROM expenses
                WHERE user_id = {PH}
                ORDER BY spent_at DESC, id DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    by_date = {}
    for r in rows:
        spent_at = r["spent_at"] if isinstance(r, sqlite3.Row) else r[0]
        item = r["item"] if isinstance(r, sqlite3.Row) else r[1]
        amount = r["amount"] if isinstance(r, sqlite3.Row) else r[2]

        d = spent_at if isinstance(spent_at, str) else spent_at.isoformat()
        by_date.setdefault(d, {"date": d, "total": 0, "expenses": []})
        by_date[d]["total"] += int(amount)
        by_date[d]["expenses"].append({"item": item, "amount": int(amount)})

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


@app.get("/change_password")
def change_password_page(request: Request):
    user = require_active(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    msg = request.query_params.get("msg", "")
    return templates.TemplateResponse(
        "change_password.html",
        {"request": request, "user": user, "msg": msg},
    )


@app.get("/")
def home(request: Request):
    init_db()
    user = get_current_user(request)
    paid_msg = request.query_params.get("paid_msg", "")
    paid_id = request.query_params.get("paid_id")
    paid_id = int(paid_id) if paid_id and paid_id.isdigit() else None

    if not user:
        return templates.TemplateResponse(
            "index.html", {"request": request, "user": None, "paid_msg": paid_msg}
        )

    master_mode = bool(user and is_master_user(user["user_id"]))

    # ✅ 收費/試用：未開通或試用到期 -> 鎖首頁功能
    access_ok, access_msg, access_until, access_mode = get_access_status(
        user["user_id"]
    )
    if not access_ok:
        group_summary = None
        # 在你算完 access_ok 之後（或至少在 return TemplateResponse 之前）
        master_mode = bool(user and is_master_user(user["user_id"]))

    # ✅ 取出該使用者所有分期資料
    rows_raw = get_all_records_for_user(user["user_id"])
    rows = [row_to_view(r) for r in rows_raw]

    paid_sum_map = get_paid_sum_map(user["user_id"])
    for r in rows:
        r["paid_sum"] = paid_sum_map.get(r["id"], 0)

    today_due_records = [r for r in rows if r["is_due_today"]]
    overdue_records = [r for r in rows if r["is_overdue"]]
    due_today_count = len(today_due_records)

    tomorrow_due_records = [
        r for r in rows if (r["days_left"] == 1) and (not r["finished"])
    ]

    today_due = len(today_due_records)
    tomorrow_due = len(tomorrow_due_records)
    overdue = len(overdue_records)

    # ✅ 今日總收 / 今日開銷
    today_total = get_today_total_for_user(user["user_id"])
    today_expense_total = get_today_expense_total_for_user(user["user_id"])
    today_expenses = get_today_expenses_for_user(user["user_id"])
    my_month_net = get_month_net_for_user(user["user_id"])

    # ✅ 今日淨收
    today_net = int(today_total) - int(today_expense_total)
    # ✅ 總票面：只計「目前資料＝未結清」(paid_count < periods) 的票面總和
    total_face_value = sum(
        int((r.get("face_value") or 0))
        for r in rows
        if int(r.get("paid_count", 0)) < int(r.get("periods", 0))
    )

    # ✅ 總票面(餘)：只計未結清資料的「票面餘」總和
    total_face_value_left = sum(
        int((r.get("face_value") or 0))
        - int((r.get("ticket_offset") or 0))
        - int((r.get("paid_sum") or 0))
        for r in rows
        if int(r.get("paid_count", 0)) < int(r.get("periods", 0))
    )

    if user and master_mode:
        group_access_ok, group_access_mode, group_access_until, group_access_msg = (
            get_group_access_status(user["user_id"])
        )
    else:
        group_access_ok, group_access_mode, group_access_until, group_access_msg = (
            False,
            "child",
            None,
            "",
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "paid_msg": paid_msg,
            # 付費/試用狀態
            "access_ok": access_ok,
            "access_until": (access_until.isoformat() if access_until else ""),
            "access_mode": access_mode,
            # 清單資料（通知/主列表會用到）
            "rows": rows,
            "today_due_records": today_due_records,
            "overdue_records": overdue_records,
            # 右欄統計
            "today_total": today_total,
            "today_expense_total": today_expense_total,
            "today_expenses": today_expenses,
            "today_net": today_net,
            "my_month_net": my_month_net,
            "total_face_value": total_face_value,
            "total_face_value_left": total_face_value_left,
            # 其他（你要留也行，不會漏）
            "paid_id": paid_id,
            "master_mode": master_mode,
            "due_today_count": due_today_count,
            "group_access_ok": group_access_ok,
            "group_access_mode": group_access_mode,
            "group_access_until": (
                group_access_until.isoformat() if group_access_until else ""
            ),
        },
    )


@app.get("/add")
def add_page(request: Request):
    user = require_active(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    return templates.TemplateResponse("add.html", {"request": request, "user": user})


@app.post("/register")
def register(
    username: str = Form(...), password: str = Form(...), display_name: str = Form(...)
):
    init_db()
    pw_hash = hash_password(password)

    # ✅ 7 天免費試用
    from datetime import timedelta

    trial_until = (date.fromisoformat(today_str()) + timedelta(days=7)).isoformat()

    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                if IS_SQLITE:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, display_name, trial_until) VALUES (?, ?, ?, ?)",
                        (username, pw_hash, display_name, trial_until),
                    )
                    user_id = cur.lastrowid
                else:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, display_name, trial_until) VALUES (%s, %s, %s, %s) RETURNING id",
                        (username, pw_hash, display_name, trial_until),
                    )
                    user_id = cur.fetchone()[0]
            conn.commit()
    except Exception:
        return RedirectResponse(
            url="/?paid_msg=" + quote("帳號已存在或註冊失敗"), status_code=303
        )

    resp = RedirectResponse(
        url="/?paid_msg=" + quote("註冊成功，已登入"), status_code=303
    )
    return set_session_cookie(
        resp,
        user_id=int(user_id),
        username=username,
        display_name=display_name,
    )


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    init_db()

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT id, password_hash, display_name FROM users WHERE username = {PH}",
                (username,),
            )

            row = cur.fetchone()

    if not row:
        return RedirectResponse(
            url="/?paid_msg=" + quote("帳號不存在"), status_code=303
        )

    user_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
    pw_hash = row["password_hash"] if isinstance(row, sqlite3.Row) else row[1]
    display_name = row["display_name"] if isinstance(row, sqlite3.Row) else row[2]

    if not verify_password(password, pw_hash):
        return RedirectResponse(url="/?paid_msg=" + quote("密碼錯誤"), status_code=303)

    resp = RedirectResponse(url="/?paid_msg=" + quote("登入成功"), status_code=303)
    return set_session_cookie(
        resp, user_id=int(user_id), username=username, display_name=display_name
    )


@app.post("/logout")
def logout():
    resp = RedirectResponse(url="/?paid_msg=" + quote("已登出"), status_code=303)
    return clear_session_cookie(resp)


@app.post("/change_password")
def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    new_password2: str = Form(...),
):
    user = require_active(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    if new_password != new_password2:
        return RedirectResponse(
            url="/change_password?msg=" + quote("兩次新密碼不一致"), status_code=303
        )

    if len(new_password) < 6:
        return RedirectResponse(
            url="/change_password?msg=" + quote("新密碼至少 6 碼"), status_code=303
        )

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT password_hash FROM users WHERE id = {PH}",
                (user["user_id"],),
            )
            row = cur.fetchone()

            if not row:
                return RedirectResponse(
                    url="/change_password?msg=" + quote("找不到帳號"), status_code=303
                )

            pw_hash = row["password_hash"] if isinstance(row, sqlite3.Row) else row[0]

            if not verify_password(current_password, pw_hash):
                return RedirectResponse(
                    url="/change_password?msg=" + quote("目前密碼錯誤"), status_code=303
                )

            cur.execute(
                f"UPDATE users SET password_hash = {PH} WHERE id = {PH}",
                (hash_password(new_password), user["user_id"]),
            )
        conn.commit()

    return RedirectResponse(
        url="/change_password?msg=" + quote("密碼更新成功"), status_code=303
    )


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
    as_period1: str | None = Form(None),
    use_face_value: str | None = Form(None),
    deduct_first_period_expense: str | None = Form(None),
    count_today: str | None = Form(None),
    ticket_deduct_one: str | None = Form(None),
):
    user = require_active(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    init_db()

    # checkbox -> bool
    as_period1_b = as_period1 == "1"
    use_face_value_b = use_face_value == "1"
    deduct_first_period_expense_b = deduct_first_period_expense == "1"
    count_today_b = count_today == "1"
    ticket_deduct_one_b = ticket_deduct_one == "1"

    face_value_effective = int(face_value or 0)

    # 你原本規則：勾「扣第一期」就把 total_amount_effective 加上一期 amount
    total_amount_effective = int(total_amount or 0)
    if deduct_first_period_expense_b:
        total_amount_effective += int(amount or 0)

    # paid_count：勾「算第一期」就視為已繳一期（維持你原本設計）
    paid_count = 1 if as_period1_b else 0
    last_paid_day = 0  # 維持你原本：建立日不算收款日

    # offset（維持你原本欄位）
    expense_offset = int(amount or 0) if ticket_deduct_one_b else 0
    ticket_offset = int(amount or 0) if use_face_value_b else 0

    # ✅ 你說「填寫支出就算入今日開銷」：支出以使用者填的 total_amount 原值為準（不含你加的一期）
    principal = 0
    try:
        principal = int(total_amount or 0)
    except Exception:
        principal = 0

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # 1) 新增 records，取得 record_id
            if IS_SQLITE:
                cur.execute(
                    f"""
                    INSERT INTO records
                    (created_date, name, face_value, total_amount, periods, amount, interval_days, paid_count, last_paid_day, user_id, ticket_offset, expense_offset)
                    VALUES ({PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH})
                    """,
                    (
                        created_date,
                        name,
                        face_value_effective,
                        total_amount_effective,
                        int(periods),
                        int(amount),
                        int(interval_days),
                        int(paid_count),
                        int(last_paid_day),
                        user["user_id"],
                        int(ticket_offset),
                        int(expense_offset),
                    ),
                )
                record_id = cur.lastrowid
            else:
                cur.execute(
                    f"""
                    INSERT INTO records
                    (created_date, name, face_value, total_amount, periods, amount, interval_days, paid_count, last_paid_day, user_id, ticket_offset, expense_offset)
                    VALUES ({PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH}, {PH})
                    RETURNING id
                    """,
                    (
                        created_date,
                        name,
                        face_value_effective,
                        total_amount_effective,
                        int(periods),
                        int(amount),
                        int(interval_days),
                        int(paid_count),
                        int(last_paid_day),
                        user["user_id"],
                        int(ticket_offset),
                        int(expense_offset),
                    ),
                )
                record_id = cur.fetchone()[0]

            # 2) ✅ 勾「是否算入今日實收」→ 寫入 payments（同一個 conn，避免外鍵炸）
            if count_today_b:
                cur.execute(
                    f"""
                    INSERT INTO payments (paid_at, amount, record_id, record_name, user_id)
                    VALUES ({PH}, {PH}, {PH}, {PH}, {PH})
                    """,
                    (
                        today_str(),
                        int(amount or 0),
                        int(record_id),
                        name,
                        user["user_id"],
                    ),
                )

            # 3) ✅ 只要「支出(total_amount)」有填且 >0 → 記一筆 expenses（算入當日開銷）
            if principal > 0:
                spent_day = created_date or today_str()
                cur.execute(
                    f"""
                    INSERT INTO expenses (spent_at, item, amount, user_id)
                    VALUES ({PH}, {PH}, {PH}, {PH})
                    """,
                    (spent_day, f"{name}（支出）", principal, user["user_id"]),
                )

        # ✅ 最後只 commit 一次
        conn.commit()

    return RedirectResponse(url="/?paid_msg=" + quote("新增完成"), status_code=303)


@app.get("/group-month")
def group_month(request: Request):
    init_db()
    user = require_active(request)
    if not user or isinstance(user, RedirectResponse):
        return user

    if not is_master_user(user["user_id"]):
        return RedirectResponse("/", status_code=303)

    msg = (request.query_params.get("msg") or "").strip()
    linked_accounts = get_linked_accounts(user["user_id"])

    return templates.TemplateResponse(
        "group_month.html",
        {
            "request": request,
            "user": user,
            "msg": msg,
            "linked_accounts": linked_accounts,
        },
    )


def get_linked_accounts(master_user_id: int) -> list[dict]:
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT al.id, u.id, u.username, u.display_name
                FROM account_links al
                JOIN users u ON u.id = al.child_user_id
                WHERE al.master_user_id = {PH}
                ORDER BY al.id DESC
                """,
                (master_user_id,),
            )
            rows = cur.fetchall()

    out = []
    for r in rows:
        if isinstance(r, sqlite3.Row):
            out.append(
                {
                    "link_id": int(r[0]),
                    "user_id": int(r[1]),
                    "username": r[2],
                    "display_name": (r[3] or ""),
                }
            )
        else:
            out.append(
                {
                    "link_id": int(r[0]),
                    "user_id": int(r[1]),
                    "username": r[2],
                    "display_name": (r[3] or ""),
                }
            )
    return out


def is_child_user(user_id: int) -> bool:
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT 1 FROM account_links WHERE child_user_id = {PH} LIMIT 1",
                (user_id,),
            )
            return cur.fetchone() is not None


def is_master_user(user_id: int) -> bool:
    # 只要不是別人的子帳號，就可當主帳號（避免「沒連過就看不到按鈕」的雞生蛋）
    return not is_child_user(user_id)


@app.get("/history")
def history(request: Request):
    user = require_active(request)
    if not user:
        return RedirectResponse(url="/?paid_msg=" + quote("請先登入"), status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    init_db()
    groups = get_history_grouped(user["user_id"])
    expense_map = get_expense_map_for_user(user["user_id"])

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "user": user,
            "groups": groups,
            "expense_map": expense_map,
        },
    )


@app.get("/admin/links", response_class=HTMLResponse)
def admin_links_page(request: Request):
    init_db()
    admin = require_admin(request)
    if not admin:
        return RedirectResponse("/?paid_msg=" + quote("無權限"), status_code=303)

    msg = request.query_params.get("msg", "")

    # 讀出所有連動
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT al.id, al.master_user_id, mu.username, mu.display_name,
                       al.child_user_id, cu.username, cu.display_name
                FROM account_links al
                JOIN users mu ON mu.id = al.master_user_id
                JOIN users cu ON cu.id = al.child_user_id
                ORDER BY al.id DESC
                """
            )
            rows = cur.fetchall()

        # sqlite row 欄位名在 join 可能不穩，簡單用 tuple 方式處理更穩
    # 改用 tuple 方式重新組一次（更穩定）
    items = []
    for r in rows:
        if isinstance(r, sqlite3.Row):
            rid = int(r[0])
            master_user_id = int(r[1])
            master_username = r[2]
            master_display = r[3]
            child_user_id = int(r[4])
            child_username = r[5]
            child_display = r[6]
        else:
            rid = int(r[0])
            master_user_id = int(r[1])
            master_username = r[2]
            master_display = r[3]
            child_user_id = int(r[4])
            child_username = r[5]
            child_display = r[6]

        items.append(
            f"""
            <tr>
              <td>{rid}</td>
              <td>{master_user_id} / {master_username} / {master_display or ""}</td>
              <td>{child_user_id} / {child_username} / {child_display or ""}</td>
              <td>
                <form method="post" action="/admin/links/delete" onsubmit="return confirm('確定刪除連動？');" style="margin:0;">
                  <input type="hidden" name="link_id" value="{rid}">
                  <button type="submit">刪除</button>
                </form>
              </td>
            </tr>
            """
        )

    html = f"""
    <html><head><meta charset="utf-8"><title>帳號連動管理</title></head>
    <body style="font-family:Arial,'Microsoft JhengHei',sans-serif; padding:16px;">
      <h2>帳號連動管理</h2>
      <p><a href="/admin/users">回使用者列表</a> ｜ <a href="/">回首頁</a></p>
      {"<p style='color:#c00;'><b>"+msg+"</b></p>" if msg else ""}

      <h3>新增連動（主帳號 -> 子帳號）</h3>
      <form method="post" action="/admin/links/add" style="display:flex; gap:10px; flex-wrap:wrap; align-items:end;">
        <label>主帳號 username<br><input name="master_username" required></label>
        <label>子帳號 username<br><input name="child_username" required></label>
        <button type="submit">新增</button>
      </form>

      <h3 style="margin-top:18px;">目前連動</h3>
      <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;">
        <tr>
          <th>ID</th><th>主帳號</th><th>子帳號</th><th>操作</th>
        </tr>
        {''.join(items) if items else '<tr><td colspan="4">尚無連動</td></tr>'}
      </table>
    </body></html>
    """
    return HTMLResponse(html)


@app.post("/admin/links/add")
def admin_links_add(
    request: Request,
    master_username: str = Form(...),
    child_username: str = Form(...),
):
    init_db()
    admin = require_admin(request)
    if not admin:
        return RedirectResponse("/?paid_msg=" + quote("無權限"), status_code=303)

    master_username = (master_username or "").strip()
    child_username = (child_username or "").strip()
    if not master_username or not child_username:
        return RedirectResponse(
            "/admin/links?msg=" + quote("請填完整"), status_code=303
        )
    if master_username == child_username:
        return RedirectResponse(
            "/admin/links?msg=" + quote("主帳號與子帳號不可相同"), status_code=303
        )

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT id FROM users WHERE username = {PH}", (master_username,)
            )
            m = cur.fetchone()
            cur.execute(
                f"SELECT id FROM users WHERE username = {PH}", (child_username,)
            )
            c = cur.fetchone()

            if not m or not c:
                return RedirectResponse(
                    "/admin/links?msg=" + quote("找不到帳號"), status_code=303
                )

            master_id = int(m["id"] if isinstance(m, sqlite3.Row) else m[0])
            child_id = int(c["id"] if isinstance(c, sqlite3.Row) else c[0])

            try:
                cur.execute(
                    f"INSERT INTO account_links (master_user_id, child_user_id) VALUES ({PH}, {PH})",
                    (master_id, child_id),
                )
                conn.commit()
            except Exception:
                # 可能重複
                if not IS_SQLITE:
                    conn.rollback()
                return RedirectResponse(
                    "/admin/links?msg=" + quote("新增失敗：可能已存在"), status_code=303
                )

    return RedirectResponse("/admin/links?msg=" + quote("新增完成"), status_code=303)


@app.post("/admin/links/delete")
def admin_links_delete(request: Request, link_id: int = Form(...)):
    init_db()
    admin = require_admin(request)
    if not admin:
        return RedirectResponse("/?paid_msg=" + quote("無權限"), status_code=303)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"DELETE FROM account_links WHERE id = {PH}", (link_id,))
        conn.commit()

    return RedirectResponse("/admin/links?msg=" + quote("已刪除"), status_code=303)


@app.get("/admin/users")
def admin_users(request: Request):
    init_db()
    admin = require_admin(request)
    if not admin:
        return RedirectResponse("/?paid_msg=" + quote("無權限"), status_code=303)

    msg = request.query_params.get("msg", "")

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                """
    SELECT id, username, display_name,
           activated_at, last_seen_at,
           group_activated_at,
           group_trial_started_at
    FROM users
    ORDER BY id DESC
"""
            )
            rows = cur.fetchall()

    users = []
    for r in rows:
        if isinstance(r, sqlite3.Row):
            users.append(
                {
                    "id": int(r["id"]),
                    "username": r["username"],
                    "display_name": r["display_name"],
                    "activated_at": r["activated_at"],
                    "last_seen_at": r["last_seen_at"],
                    "group_activated_at": r["group_activated_at"],
                    "group_trial_started_at": r["group_trial_started_at"],
                }
            )

        else:
            users.append(
                {
                    "id": int(r[0]),
                    "username": r[1],
                    "display_name": r[2],
                    "activated_at": r[3],
                    "last_seen_at": r[4],
                    "group_activated_at": r[5],
                    "group_trial_started_at": r[6],
                }
            )

    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "user": admin, "users": users, "msg": msg},
    )


@app.post("/admin/users/activate")
def admin_users_activate(
    request: Request,
    user_id: int = Form(...),
    days: str = Form(""),
    clear: str | None = Form(None),
):
    init_db()
    admin = require_admin(request)
    if not admin:
        return RedirectResponse("/?paid_msg=" + quote("無權限"), status_code=303)

    msg = request.query_params.get("msg", "")

    # 清除開通時間
    if clear == "1":
        activated_at = ""

    from datetime import date, timedelta

    days = (days or "").strip()

    # 清除開通
    if clear == "1":
        value = None
    else:
        if not days:
            value = None
        else:
            d = int(days)  # 例如 30
            until = date.today() + timedelta(days=d)  # 你要 A 版本：今天+30天
            value = until.isoformat()  # "2026-02-17" 這種

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"UPDATE users SET activated_at = {PH} WHERE id = {PH}",
                (value, user_id),
            )
        conn.commit()

    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/create")
def admin_users_create(
    request: Request,
    display_name: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
):
    init_db()
    admin = require_admin(request)
    if not admin:
        return RedirectResponse("/?paid_msg=" + quote("無權限"), status_code=303)

    msg = request.query_params.get("msg", "")

    display_name = (display_name or "").strip()
    username = (username or "").strip()
    password = (password or "").strip()

    if not display_name or not username or not password:
        return RedirectResponse(
            "/admin/users?msg=" + quote("請填：名稱 / 帳號 / 密碼"), status_code=303
        )

    if len(password) < 6:
        return RedirectResponse(
            "/admin/users?msg=" + quote("密碼至少 6 碼"), status_code=303
        )

    # ✅ 7 天免費試用
    from datetime import timedelta

    trial_until = (date.fromisoformat(today_str()) + timedelta(days=7)).isoformat()
    pw_hash = hash_password(password)

    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                if IS_SQLITE:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, display_name, trial_until) VALUES (?, ?, ?, ?)",
                        (username, pw_hash, display_name, trial_until),
                    )
                else:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, display_name, trial_until) VALUES (%s, %s, %s, %s)",
                        (username, pw_hash, display_name, trial_until),
                    )
            conn.commit()
    except Exception:
        return RedirectResponse(
            "/admin/users?msg=" + quote("建立失敗：帳號可能已存在"), status_code=303
        )

    return RedirectResponse("/admin/users?msg=" + quote("已新增帳號"), status_code=303)


@app.post("/admin/users/delete")
def admin_users_delete(
    request: Request,
    user_id: int = Form(...),
):
    init_db()
    admin = require_admin(request)
    if not admin:
        return RedirectResponse("/?paid_msg=" + quote("無權限"), status_code=303)

    msg = request.query_params.get("msg", "")

    # 防止刪到自己
    if int(user_id) == int(admin.get("user_id")):
        return RedirectResponse(
            "/admin/users?msg=" + quote("不能刪除自己的帳號"), status_code=303
        )

    # 防止刪除管理員帳號（用 username 判斷）
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"SELECT username FROM users WHERE id = {PH}", (user_id,))
            row = cur.fetchone()
            target_username = (
                row[0]
                if (row and not isinstance(row, sqlite3.Row))
                else (row["username"] if row else "")
            )

    if ADMIN_USERNAME and target_username == ADMIN_USERNAME:
        return RedirectResponse(
            "/admin/users?msg=" + quote("不能刪除管理員帳號"), status_code=303
        )

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(f"DELETE FROM users WHERE id = {PH}", (user_id,))
        conn.commit()

    return RedirectResponse("/admin/users?msg=" + quote("已刪除帳號"), status_code=303)


@app.post("/history/delete")
def delete_payment(request: Request, payment_id: int = Form(...)):
    init_db()
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                DELETE FROM payments
                WHERE id = {PH} AND user_id = {PH}
            """,
                (payment_id, user["user_id"]),
            )
        conn.commit()

    return RedirectResponse("/history", status_code=303)


@app.post("/group/link/add")
def group_link_add(
    request: Request,
    child_username: str = Form(...),
    child_password: str = Form(...),
):
    init_db()
    user = require_active(request)
    if not user or isinstance(user, RedirectResponse):
        return user

    if not is_master_user(user["user_id"]):
        return RedirectResponse(
            "/?paid_msg=" + quote("子帳號無法使用連結功能"), status_code=303
        )

    child_username = (child_username or "").strip()
    child_password = (child_password or "").strip()
    if not child_username or not child_password:
        return RedirectResponse(
            "/group-month?msg=" + quote("請輸入子帳號與密碼"), status_code=303
        )

    if child_username == user["username"]:
        return RedirectResponse(
            "/group-month?msg=" + quote("不能連結自己"), status_code=303
        )

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT id, password_hash FROM users WHERE username = {PH}",
                (child_username,),
            )
            row = cur.fetchone()
            if not row:
                return RedirectResponse(
                    "/group-month?msg=" + quote("找不到子帳號"), status_code=303
                )

            child_id = int(row["id"] if isinstance(row, sqlite3.Row) else row[0])
            pw_hash = row["password_hash"] if isinstance(row, sqlite3.Row) else row[1]

            if not verify_password(child_password, pw_hash):
                return RedirectResponse(
                    "/group-month?msg=" + quote("子帳號密碼錯誤"), status_code=303
                )

            # 子帳號已被別人連結就擋
            cur.execute(
                f"SELECT 1 FROM account_links WHERE child_user_id = {PH} LIMIT 1",
                (child_id,),
            )
            if cur.fetchone():
                return RedirectResponse(
                    "/group-month?msg=" + quote("此子帳號已被其他主帳號連結"),
                    status_code=303,
                )

            try:
                cur.execute(
                    f"INSERT INTO account_links (master_user_id, child_user_id) VALUES ({PH}, {PH})",
                    (user["user_id"], child_id),
                )
                conn.commit()
            except Exception:
                if not IS_SQLITE:
                    conn.rollback()
                return RedirectResponse(
                    "/group-month?msg=" + quote("新增失敗：可能已存在"), status_code=303
                )

    return RedirectResponse("/group-month?msg=" + quote("連結完成"), status_code=303)


@app.post("/group/link/delete")
def group_link_delete(request: Request, link_id: int = Form(...)):
    init_db()
    user = require_active(request)
    if not user or isinstance(user, RedirectResponse):
        return user

    if not is_master_user(user["user_id"]):
        return RedirectResponse(
            "/?paid_msg=" + quote("子帳號無法使用連結功能"), status_code=303
        )

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # 只能刪自己的連結
            cur.execute(
                f"DELETE FROM account_links WHERE id = {PH} AND master_user_id = {PH}",
                (link_id, user["user_id"]),
            )
        conn.commit()

    return RedirectResponse("/group-month?msg=" + quote("已解除連結"), status_code=303)


@app.post("/admin/users/group_activate")
def admin_users_group_activate(
    request: Request,
    user_id: int = Form(...),
    days: str = Form(""),
    clear: str | None = Form(None),
):
    init_db()
    admin = require_admin(request)
    if not admin:
        return RedirectResponse("/?paid_msg=" + quote("無權限"), status_code=303)

    from datetime import date, timedelta

    days = (days or "").strip()

    if clear == "1":
        value = None
    else:
        if not days:
            value = None
        else:
            d = int(days)
            until = date.fromisoformat(today_str()) + timedelta(
                days=d
            )  # ✅ 用台北日期一致
            value = until.isoformat()

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"UPDATE users SET group_activated_at = {PH} WHERE id = {PH}",
                (value, user_id),
            )
        conn.commit()

    return RedirectResponse("/admin/users", status_code=303)


@app.get("/add-page")
def add_page_2(request: Request):
    user = require_active(request)
    if not user:
        return RedirectResponse(url="/", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    return templates.TemplateResponse("add.html", {"request": request, "user": user})


@app.get("/expense-page")
def expense_page(request: Request):
    user = require_active(request)
    if not user:
        return RedirectResponse(url="/", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    return templates.TemplateResponse(
        "expense_add.html", {"request": request, "user": user}
    )


@app.post("/expense/add")
def add_expense(
    request: Request,
    item: list[str] = Form([]),
    amount: list[str] = Form([]),
):
    user = require_active(request)
    if not user:
        return RedirectResponse(url="/", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    init_db()

    items = item or []
    amounts = amount or []

    to_insert: list[tuple[str, int]] = []
    for it, am in zip(items, amounts):
        it = (it or "").strip()
        try:
            am_int = int(am)
        except Exception:
            am_int = 0
        if it and am_int > 0:
            to_insert.append((it, am_int))

    if not to_insert:
        return RedirectResponse(url="/expense-page", status_code=303)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            for it, am_int in to_insert:
                cur.execute(
                    f"INSERT INTO expenses (spent_at, item, amount, user_id) VALUES ({PH}, {PH}, {PH}, {PH})",
                    (today_str(), it, am_int, user["user_id"]),
                )
        conn.commit()

    return RedirectResponse(
        url="/?paid_msg=" + quote(f"新增開銷完成（{len(to_insert)}筆）"),
        status_code=303,
    )


@app.get("/expense/edit/{expense_id}")
def expense_edit_page(request: Request, expense_id: int):
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    init_db()
    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"SELECT id, item, amount FROM expenses WHERE id = {PH} AND user_id = {PH}",
                (expense_id, user["user_id"]),
            )
            row = cur.fetchone()

    if not row:
        return RedirectResponse("/", status_code=303)

    if isinstance(row, sqlite3.Row):
        e = {"id": int(row["id"]), "item": row["item"], "amount": int(row["amount"])}
    else:
        e = {"id": int(row[0]), "item": row[1], "amount": int(row[2])}

    return templates.TemplateResponse(
        "expense_edit.html", {"request": request, "user": user, "e": e}
    )


@app.post("/expense/delete-multiple")
def delete_expenses_multiple(
    request: Request,
    expense_ids: list[str] = Form([]),
):
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    init_db()

    # 轉成 int，避免亂值
    ids: list[int] = []
    for x in expense_ids or []:
        try:
            ids.append(int(x))
        except Exception:
            pass

    if not ids:
        return RedirectResponse("/", status_code=303)

    placeholders = ",".join([PH] * len(ids))
    params = [user["user_id"], *ids]

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                DELETE FROM expenses
                WHERE user_id = {PH}
                  AND id IN ({placeholders})
                """,
                params,
            )
        conn.commit()

    return RedirectResponse("/?paid_msg=已刪除開銷", status_code=303)


@app.post("/expense/edit/{expense_id}")
def expense_edit_save(
    request: Request,
    expense_id: int,
    item: str = Form(...),
    amount: int = Form(...),
):
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    init_db()
    item = (item or "").strip()
    if not item or amount <= 0:
        return RedirectResponse(f"/expense/edit/{expense_id}", status_code=303)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                UPDATE expenses
                SET item = {PH}, amount = {PH}
                WHERE id = {PH} AND user_id = {PH}
                """,
                (item, amount, expense_id, user["user_id"]),
            )
        conn.commit()

    return RedirectResponse("/?paid_msg=" + quote("開銷已更新"), status_code=303)


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/pay/{record_id}")
def pay_record(
    request: Request,
    record_id: int,
    periods: str = Form("1"),  # 右邊逾期補繳會傳 periods；今日已繳款沒傳就預設 1
):
    init_db()
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    # 防呆
    try:
        periods = int(periods)
    except Exception:
        periods = 1
    if periods <= 0:
        periods = 1

    # 取這筆資料（確保是自己的）
    r = get_record_for_user(record_id, user["user_id"])
    if not r:
        return RedirectResponse("/?paid_msg=" + quote("找不到資料"), status_code=303)

    # 已結清就不處理
    if r["paid_count"] >= r["periods"]:
        return RedirectResponse("/?paid_msg=" + quote("此筆已結清"), status_code=303)

    remaining = r["periods"] - r["paid_count"]
    if remaining < 0:
        remaining = 0

    # 不能補繳超過剩餘期數
    if periods > remaining:
        periods = remaining
    if periods <= 0:
        return RedirectResponse("/?paid_msg=" + quote("沒有可繳期數"), status_code=303)

    pay_amount = periods * r["amount"]

    # 更新 last_paid_day：用「目前已過天數」做基準，避免亂跳
    created_date_obj = date.fromisoformat(r["created_date"])
    current_day = calc_current_day(created_date_obj)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # 寫 payments（記一筆：今天收款）
            cur.execute(
                f"""
    INSERT INTO payments (paid_at, amount, record_id, record_name, user_id)
    VALUES ({PH}, {PH}, {PH}, {PH}, {PH})
    """,
                (today_str(), pay_amount, record_id, r["name"], user["user_id"]),
            )

            # 更新 records 的已繳期數 & 最後繳款日
            cur.execute(
                f"""
                UPDATE records
                SET paid_count = paid_count + {PH},
                    last_paid_day = {PH}
                WHERE id = {PH} AND user_id = {PH}
                """,
                (periods, current_day, record_id, user["user_id"]),
            )

        conn.commit()

    return RedirectResponse(
        url="/?paid_id=" + quote(str(record_id)) + "&paid_msg=" + quote("繳款完成"),
        status_code=303,
    )


@app.post("/settle/{record_id}")
def settle_record(request: Request, record_id: int, amount: int = Form(...)):
    init_db()
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    if amount <= 0:
        return RedirectResponse("/", status_code=303)

    # ✅ 先拿到這筆資料，算 current_day（放在 DB 操作前也可以）
    r = get_record_for_user(record_id, user["user_id"])
    if not r:
        return RedirectResponse("/", status_code=303)

    created_date_obj = date.fromisoformat(r["created_date"])
    current_day = calc_current_day(created_date_obj)

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # 寫一筆 payments（結清）
            cur.execute(
                f"""
    INSERT INTO payments (paid_at, amount, record_id, record_name, user_id)
    VALUES ({PH}, {PH}, {PH}, {PH}, {PH})
    """,
                (today_str(), amount, record_id, r["name"], user["user_id"]),
            )

            # ✅ 標記完成 + last_paid_day 用 current_day
            cur.execute(
                f"""
                UPDATE records
                SET paid_count = periods,
                    last_paid_day = {PH}
                WHERE id = {PH} AND user_id = {PH}
                """,
                (current_day, record_id, user["user_id"]),
            )

        conn.commit()

    return RedirectResponse("/", status_code=303)


@app.post("/history/delete-multiple")
def delete_payments_multiple(request: Request, payment_ids: list[str] = Form([])):
    init_db()
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    # ✅ 轉 int（避免 FastAPI 解析 list[int] 時炸）
    _ids = []
    for x in payment_ids or []:
        try:
            _ids.append(int(x))
        except Exception:
            pass
    payment_ids = _ids

    if not payment_ids:
        return RedirectResponse("/history", status_code=303)

    placeholders = ",".join([PH] * len(payment_ids))
    params = [user["user_id"], *payment_ids]

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"DELETE FROM payments WHERE user_id = {PH} AND id IN ({placeholders})",
                params,
            )
        conn.commit()

    return RedirectResponse("/history", status_code=303)


@app.post("/delete/{record_id}")
def delete_record(request: Request, record_id: int):
    init_db()
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # ✅ 不刪 payments、不刪 records，只標記刪除
            cur.execute(
                f"UPDATE records SET is_deleted = TRUE WHERE id = {PH} AND user_id = {PH}",
                (record_id, user["user_id"]),
            )
        conn.commit()

    return RedirectResponse("/", status_code=303)


@app.post("/delete-multiple")
def delete_multiple(request: Request, record_ids: list[str] = Form([])):
    init_db()
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    ids: list[int] = []
    for x in record_ids or []:
        try:
            ids.append(int(x))
        except Exception:
            pass

    if not ids:
        return RedirectResponse("/", status_code=303)

    placeholders = ",".join([PH] * len(ids))
    params = [user["user_id"], *ids]

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            # 如果你的 payments 有 ON DELETE CASCADE，這段可以留著或不留都行
            cur.execute(
                f"DELETE FROM payments WHERE user_id = {PH} AND record_id IN ({placeholders})",
                params,
            )
            cur.execute(
                f"DELETE FROM records WHERE user_id = {PH} AND id IN ({placeholders})",
                params,
            )
        conn.commit()

    return RedirectResponse("/", status_code=303)


@app.get("/edit/{record_id}")
def edit_page(request: Request, record_id: int):
    init_db()
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT id, created_date, name, face_value, total_amount, periods, amount,
       interval_days, paid_count, last_paid_day, user_id, ticket_offset, expense_offset
FROM records
WHERE id = {PH} AND user_id = {PH}
            """,
                (record_id, user["user_id"]),
            )
            row = cur.fetchone()
            if not row:
                return RedirectResponse("/", status_code=303)

    r = row_to_view(row)
    return templates.TemplateResponse(
        "edit.html", {"request": request, "user": user, "r": r}
    )


@app.post("/edit/{record_id}")
def edit_save(
    request: Request,
    record_id: int,
    created_date: str = Form(...),
    name: str = Form(...),
    face_value: int = Form(0),
    total_amount: int = Form(...),
    periods: int = Form(...),
    amount: int = Form(...),
    interval_days: int = Form(...),
):
    init_db()
    user = require_active(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if isinstance(user, RedirectResponse):
        return user

    with get_conn() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                f"""
                UPDATE records
                SET created_date = {PH},
                    name = {PH},
                    face_value = {PH},
                    total_amount = {PH},
                    periods = {PH},
                    amount = {PH},
                    interval_days = {PH}
                WHERE id = {PH} AND user_id = {PH}
            """,
                (
                    created_date,
                    name,
                    face_value,
                    total_amount,
                    periods,
                    amount,
                    interval_days,
                    record_id,
                    user["user_id"],
                ),
            )
        conn.commit()

    return RedirectResponse("/", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
