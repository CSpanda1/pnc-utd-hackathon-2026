"""Single-process teaching API: opaque sessions, SQLite, synthetic customer isolation."""
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import threading
import time
from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator
from pwdlib import PasswordHash

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB))
from financial_tools import (calculate_indicators, get_accounts, simulate_purchase, get_transactions,
                             get_account_history, summarize_transaction_history, get_history_evidence)
from llm_demo import narrate

PASSWORDS = PasswordHash.recommended()
DUMMY_HASH = PASSWORDS.hash(secrets.token_urlsafe(32))
COOKIE = 'wellness_session'
SESSION_SECONDS = 3600


@dataclass(frozen=True)
class Settings:
    database: Path
    frontend_origin: str = 'http://localhost:3000'
    secure_cookie: bool = False
    llm_provider: str = 'none'

    @classmethod
    def environment(cls):
        return cls(Path(os.getenv('DATABASE_PATH', str(Path(__file__).parent / 'state/workshop.sqlite3'))),
                   os.getenv('FRONTEND_ORIGIN', 'http://localhost:3000'),
                   os.getenv('COOKIE_SECURE', '0') == '1', os.getenv('LLM_PROVIDER', 'none'))


class Credentials(BaseModel):
    model_config = ConfigDict(extra='forbid')
    email: EmailStr = Field(max_length=254)
    password: str = Field(min_length=12, max_length=128)


class Scenario(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    purchase_price: float = Field(default=30000, ge=0, le=150000)
    down_payment: float = Field(default=5000, ge=0, le=150000)
    apr_pct: float = Field(default=8, ge=0, le=35)
    term_months: int = Field(default=60, ge=6, le=120, strict=True)

    @model_validator(mode='after')
    def check_down_payment(self):
        if self.down_payment > self.purchase_price:
            raise ValueError('Down payment cannot exceed price')
        return self


class TransactionFilters(BaseModel):
    model_config = ConfigDict(extra='forbid')
    account_id: str | None = Field(default=None, min_length=1, max_length=64)
    start_date: date | None = None
    end_date: date | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def ordered_dates(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError('Start date must not follow end date')
        return self


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def create_app(settings=None):
    settings = settings or Settings.environment()
    if settings.llm_provider not in {'none', 'huggingface', 'foundry', 'openai'}:
        raise ValueError('Web demo LLM_PROVIDER must be none, huggingface, foundry, or openai')
    if settings.secure_cookie and not settings.frontend_origin.startswith('https://'):
        raise ValueError('Secure cookies require an HTTPS FRONTEND_ORIGIN')
    if not settings.secure_cookie and settings.frontend_origin not in {
        'http://localhost:3000', 'http://127.0.0.1:3000'
    }:
        raise ValueError('Use COOKIE_SECURE=1 and HTTPS outside localhost')

    @contextmanager
    def database():
        conn = sqlite3.connect(settings.database, timeout=15)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA foreign_keys=ON')
            with conn:
                yield conn
        finally:
            conn.close()

    @asynccontextmanager
    async def lifespan(app):
        settings.database.parent.mkdir(parents=True, exist_ok=True)
        with database() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL, customer_id TEXT UNIQUE NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                    expires_at INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS rate_limits (
                    bucket TEXT PRIMARY KEY, attempts INTEGER NOT NULL, reset_at INTEGER NOT NULL);
            ''')
        yield

    app = FastAPI(title='Synthetic Financial Wellness API', lifespan=lifespan)
    inference_lock = threading.Lock()

    @app.middleware('http')
    async def security_headers(request: Request, call_next):
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            if (request.headers.get('origin') != settings.frontend_origin
                    or request.headers.get('x-workshop-csrf') != '1'):
                return Response('Origin or CSRF header rejected', status_code=403,
                                headers={'Cache-Control': 'no-store'})
            if request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return Response('JSON required', status_code=415)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    def rate_limit(bucket, limit, seconds):
        now = int(time.time())
        with database() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('DELETE FROM rate_limits WHERE reset_at<=?', (now,))
            row = conn.execute('SELECT * FROM rate_limits WHERE bucket=?', (bucket,)).fetchone()
            if row and row['attempts'] >= limit:
                raise HTTPException(429, 'Too many attempts; try again later',
                                    headers={'Retry-After': str(row['reset_at'] - now)})
            conn.execute('''INSERT INTO rate_limits VALUES (?,1,?)
                ON CONFLICT(bucket) DO UPDATE SET attempts=attempts+1''', (bucket, now + seconds))

    def auth_limit(request, email, action):
        # Never trust client-supplied X-Forwarded-For. With the Next.js proxy, the
        # peer limit is shared by the classroom; email limits remain independent.
        peer = request.client.host if request.client else 'unknown'
        rate_limit('auth-peer:' + digest(peer), 120, 600)
        rate_limit(action + ':' + digest(email), 10, 600)

    def user_view(user):
        return {key: user[key] for key in ('id', 'email', 'customer_id')}

    def issue_session(conn, user_id, response):
        token = secrets.token_urlsafe(32)
        conn.execute('DELETE FROM sessions WHERE expires_at<=?', (int(time.time()),))
        conn.execute('INSERT INTO sessions VALUES (?,?,?)',
                     (digest(token), user_id, int(time.time()) + SESSION_SECONDS))
        response.set_cookie(COOKIE, token, httponly=True, secure=settings.secure_cookie,
                            samesite='strict', max_age=SESSION_SECONDS, path='/')

    def current_user(request: Request):
        token = request.cookies.get(COOKIE)
        if not token or len(token) > 128:
            raise HTTPException(401, 'Sign in required')
        with database() as conn:
            user = conn.execute('''SELECT users.* FROM users JOIN sessions ON users.id=sessions.user_id
                WHERE sessions.token_hash=? AND sessions.expires_at>?''',
                                (digest(token), int(time.time()))).fetchone()
        if not user:
            raise HTTPException(401, 'Session expired; sign in again')
        return user

    User = Annotated[sqlite3.Row, Depends(current_user)]

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.post('/auth/register', status_code=201)
    def register(body: Credentials, request: Request, response: Response):
        email = str(body.email).casefold()
        auth_limit(request, email, 'register')
        password_hash = PASSWORDS.hash(body.password)
        with database() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute('SELECT 1 FROM users WHERE email=?', (email,)).fetchone():
                raise HTTPException(409, 'Registration unavailable for this email')
            allocated = {row[0] for row in conn.execute('SELECT customer_id FROM users')}
            customer_id = next((f'C{i:03d}' for i in range(1, 31)
                                if f'C{i:03d}' not in allocated), None)
            if not customer_id:
                raise HTTPException(409, 'All 30 workshop profiles are allocated')
            cursor = conn.execute('INSERT INTO users(email,password_hash,customer_id) VALUES (?,?,?)',
                                  (email, password_hash, customer_id))
            # Registration rotates any existing browser session.
            conn.execute('DELETE FROM sessions WHERE token_hash=?',
                         (digest(request.cookies.get(COOKIE, '')),))
            issue_session(conn, cursor.lastrowid, response)
            return {'id': cursor.lastrowid, 'email': email, 'customer_id': customer_id}

    @app.post('/auth/login')
    def login(body: Credentials, request: Request, response: Response):
        email = str(body.email).casefold()
        auth_limit(request, email, 'login')
        with database() as conn:
            user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
            valid = PASSWORDS.verify(body.password, user['password_hash'] if user else DUMMY_HASH)
            if not valid or not user:
                raise HTTPException(401, 'Invalid email or password')
            conn.execute('DELETE FROM sessions WHERE token_hash=?',
                         (digest(request.cookies.get(COOKIE, '')),))
            issue_session(conn, user['id'], response)
            return user_view(user)

    @app.post('/auth/logout', status_code=204)
    def logout(request: Request):
        with database() as conn:
            conn.execute('DELETE FROM sessions WHERE token_hash=?',
                         (digest(request.cookies.get(COOKIE, '')),))
        response = Response(status_code=204)
        response.delete_cookie(COOKIE, path='/', httponly=True,
                               secure=settings.secure_cookie, samesite='strict')
        return response

    @app.get('/me')
    def me(user: User):
        return user_view(user)

    @app.get('/financials')
    def financials(user: User):
        return {'indicators': calculate_indicators(user['customer_id']),
                'accounts': get_accounts(user['customer_id']),
                'history': summarize_transaction_history(user['customer_id']),
                'account_history': get_account_history(user['customer_id'])}

    @app.get('/transactions')
    def transactions(user: User, filters: Annotated[TransactionFilters, Query()]):
        if filters.account_id and not any(a['account_id'] == filters.account_id for a in get_accounts(user['customer_id'])):
            raise HTTPException(404, 'Account not available for this customer')
        return get_transactions(user['customer_id'], filters.account_id,
                                filters.start_date.isoformat() if filters.start_date else None,
                                filters.end_date.isoformat() if filters.end_date else None,
                                filters.limit, filters.offset)

    @app.post('/scenario')
    def scenario(body: Scenario, user: User):
        return simulate_purchase(user['customer_id'], **body.model_dump())

    @app.post('/narration')
    def narration(body: Scenario, user: User):
        rate_limit('narration:' + str(user['id']), 10, 3600)
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(503, 'Model is busy; try again shortly', headers={'Retry-After': '10'})
        try:
            evidence = {'indicators': calculate_indicators(user['customer_id']),
                        'scenario': simulate_purchase(user['customer_id'], **body.model_dump()),
                        'transaction_history': get_history_evidence(user['customer_id'])}
            return {'provider': settings.llm_provider, 'answer': narrate(evidence, settings.llm_provider)}
        except (ImportError, RuntimeError, OSError, ValueError):
            raise HTTPException(503, 'Local narration unavailable; check server model configuration') from None
        finally:
            inference_lock.release()

    return app


app = create_app()
