"""One durable campaign ledger, including reservations made before HTTP sends."""
import contextlib
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid



class BudgetStop(RuntimeError):
    pass


class BudgetBusy(RuntimeError):
    """No request sent; wait for another live experiment's request slot."""


def money(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected nonnegative finite money")
    return float(value)


class Ledger:
    LIMIT = 100.0
    PARALLEL_REQUESTS = 2

    def __init__(self, path, *, parallel=False, parallel_requests=None):
        self.path = Path(path).resolve()
        self.parallel = parallel
        self.parallel_requests = self.PARALLEL_REQUESTS if parallel_requests is None else parallel_requests
        if type(self.parallel_requests) is not int or not 1 <= self.parallel_requests <= 10:
            raise ValueError('parallel_requests must be between 1 and 10')
        self._leases = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, cap REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (
                  id TEXT PRIMARY KEY, run TEXT NOT NULL, phase TEXT, agent TEXT,
                  model TEXT, reserved REAL NOT NULL, actual REAL, status TEXT NOT NULL,
                  receipt TEXT, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS campaign_settings (key TEXT PRIMARY KEY, value REAL NOT NULL);
                INSERT OR IGNORE INTO campaign_settings VALUES ('ceiling_usd', 100.0);
                CREATE TABLE IF NOT EXISTS budget_amendments (
                  old_limit REAL NOT NULL, new_limit REAL NOT NULL, reason TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS run_budget_amendments (
                  run TEXT NOT NULL, old_cap REAL NOT NULL, new_cap REAL NOT NULL,
                  reason TEXT NOT NULL, created REAL NOT NULL);
            """)

    @property
    def limit(self):
        with self.connect() as db:
            return db.execute("SELECT value FROM campaign_settings WHERE key='ceiling_usd'").fetchone()[0]

    def amend_limit(self, new_limit, reason):
        """Operator-only budget amendment after explicit user authorization."""
        new_limit = money(new_limit)
        if new_limit <= 0 or not isinstance(reason, str) or len(reason.strip()) < 20:
            raise ValueError("A positive limit and concrete authorization reason are required")
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            accounted = math.fsum(r[0] for r in db.execute('SELECT COALESCE(actual,reserved) FROM calls'))
            if new_limit < accounted:
                raise ValueError("New ceiling cannot be below accounted spending and reservations")
            old = db.execute("SELECT value FROM campaign_settings WHERE key='ceiling_usd'").fetchone()[0]
            db.execute("UPDATE campaign_settings SET value=? WHERE key='ceiling_usd'", (new_limit,))
            db.execute('INSERT INTO budget_amendments VALUES (?,?,?,?)', (old, new_limit, reason.strip(), time.time()))
        return {'old_limit': old, 'new_limit': new_limit, 'accounted_usd': accounted, 'reason': reason.strip()}

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def register(self, run, cap):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            ceiling = db.execute("SELECT value FROM campaign_settings WHERE key='ceiling_usd'").fetchone()[0]
            if not 0 < money(cap) <= ceiling:
                raise ValueError("Run cap must be positive and within the authorized campaign ceiling")
            db.execute("INSERT INTO runs VALUES (?,?)", (run, cap))
        if self.parallel:
            path = self._lease_path(run)
            path.parent.mkdir(exist_ok=True)
            lease = path.open('a+b')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                lease.close()
                raise
            self._leases[run] = lease

    def _lease_path(self, run):
        name = hashlib.sha256(run.encode()).hexdigest()
        return self.path.parent / 'request-leases' / (name + '.lock')

    def amend_run_cap(self, run, cap, reason):
        """Operator amendment; preserve every charge/hold and wait for settlement."""
        cap = money(cap)
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            raise ValueError("A concrete authorization reason is required")
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT cap FROM runs WHERE id=?', (run,)).fetchone()
            if row is None:
                raise ValueError("Unknown episode")
            calls = db.execute('SELECT * FROM calls WHERE run=?', (run,)).fetchall()
            if any(c['status'] in {'pending', 'parallel_pending'} for c in calls):
                raise ValueError("Wait for pending calls to settle before changing the episode cap")
            accounted = math.fsum(c['actual'] if c['actual'] is not None else c['reserved'] for c in calls)
            ceiling = db.execute("SELECT value FROM campaign_settings WHERE key='ceiling_usd'").fetchone()[0]
            if cap < accounted or cap > ceiling:
                raise ValueError("Cap must cover existing charges/holds and fit the campaign ceiling")
            db.execute('UPDATE runs SET cap=? WHERE id=?', (cap, run))
            db.execute('INSERT INTO run_budget_amendments VALUES (?,?,?,?,?)',
                       (run, row['cap'], cap, reason.strip(), time.time()))
        return {'run': run, 'old_cap': row['cap'], 'new_cap': cap, 'accounted_usd': accounted,
                'reason': reason.strip()}

    def attach(self, run, cap):
        """Continue an audited episode without resetting its spent or reserved cap."""
        if not self.parallel:
            raise ValueError('Recovery requires a live parallel request lease')
        with self.connect() as db:
            row = db.execute('SELECT cap FROM runs WHERE id=?', (run,)).fetchone()
        if row is None or not math.isclose(row['cap'], money(cap), abs_tol=1e-12):
            raise ValueError('Original episode cap must match exactly')
        if self.summary(run)['pending'] or self.summary(run)['blocked'] or self.summary()['blocked']:
            raise BudgetStop('Outstanding request requires accounting review before continuation')
        path = self._lease_path(run); path.parent.mkdir(exist_ok=True)
        lease = path.open('a+b')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            lease.close(); raise
        self._leases[run] = lease

    def _owner_alive(self, run):
        path = self._lease_path(run)
        if not path.exists():
            return False
        with path.open('rb') as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lease, fcntl.LOCK_UN)
            return False

    def recover_orphans(self):
        """Keep abandoned reservations in full and block further spending."""
        orphaned = []
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT id,run FROM calls WHERE status='parallel_pending'").fetchall():
                if not self._owner_alive(row['run']):
                    db.execute("UPDATE calls SET status='uncertain' WHERE id=?", (row['id'],))
                    orphaned.append(row['id'])
        return orphaned

    def close(self):
        for lease in self._leases.values():
            lease.close()
        self._leases.clear()
        self.recover_orphans()

    def summary(self, run=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM calls" + (" WHERE run=?" if run else ""),
                              (run,) if run else ()).fetchall()
            ceiling = db.execute("SELECT value FROM campaign_settings WHERE key='ceiling_usd'").fetchone()[0]
        actual = math.fsum(r["actual"] or 0 for r in rows)
        reserved = math.fsum(r["reserved"] for r in rows if r["actual"] is None)
        return {"reported_usd": actual, "unconfirmed_usd": reserved,
                "accounted_usd": actual + reserved, "calls": len(rows),
                "ceiling_usd": ceiling,
                "blocked": any(r["status"] in {"uncertain", "overrun"} or
                               (run is not None and r["status"] == "quarantined_timeout") for r in rows),
                "quarantined_timeouts": sum(r["status"] == "quarantined_timeout" for r in rows),
                "pending": sum(r["status"] in {"pending", "parallel_pending"} for r in rows)}

    def reserve(self, run, amount, phase, agent, model):
        amount = money(amount)
        if (self.path.parent / "STOP").exists():
            raise BudgetStop("Campaign STOP file exists")
        self.recover_orphans()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT * FROM calls").fetchall()
            if any(r["status"] in {"uncertain", "overrun"} for r in rows):
                raise BudgetStop("Unreconciled prior call; reservation retained, no automatic replay")
            if any(r['run'] == run and r['status'] == 'quarantined_timeout' for r in rows):
                raise BudgetStop("This episode timed out; full charge held and further requests blocked")
            if self.parallel:
                if run not in self._leases:
                    raise BudgetStop('Parallel experiment does not own its live request lease')
                # Legacy workers already in progress recognize pending, uncertain,
                # and overrun. They count every NULL actual as reserved, including
                # parallel_pending. A distinct live status lets them continue
                # unchanged without weakening either shared dollar limit.
                if any(r['status'] == 'pending' for r in rows):
                    raise BudgetBusy('Waiting for the existing serial request to settle')
                if sum(r['status'] == 'parallel_pending' for r in rows) >= self.parallel_requests:
                    raise BudgetBusy('Parallel request slots are occupied')
                status = 'parallel_pending'
            else:
                if any(r['status'] in {'pending', 'parallel_pending'} for r in rows):
                    raise BudgetStop('Unreconciled prior call; reservation retained, no automatic replay')
                status = 'pending'
            total = math.fsum(r["actual"] if r["actual"] is not None else r["reserved"] for r in rows)
            local = math.fsum(r["actual"] if r["actual"] is not None else r["reserved"]
                              for r in rows if r["run"] == run)
            cap = db.execute("SELECT cap FROM runs WHERE id=?", (run,)).fetchone()
            ceiling = db.execute("SELECT value FROM campaign_settings WHERE key='ceiling_usd'").fetchone()[0]
            if not cap or total + amount > ceiling or local + amount > cap["cap"]:
                raise BudgetStop("Next call exceeds campaign or episode budget")
            ident = uuid.uuid4().hex
            db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,NULL,?,NULL,?)",
                       (ident, run, phase, agent, model, amount, status, time.time()))
        return ident

    def settle(self, ident, actual, receipt):
        actual = money(actual)
        with self.connect() as db:
            row = db.execute("SELECT * FROM calls WHERE id=?", (ident,)).fetchone()
            if not row or row["status"] not in {"pending", "parallel_pending"}:
                raise ValueError("Call is not pending")
            status = "overrun" if actual > row["reserved"] + 1e-9 else "charged"
            db.execute("UPDATE calls SET actual=?,status=?,receipt=? WHERE id=?",
                       (actual, status, json.dumps(receipt), ident))
        if status == "overrun":
            raise BudgetStop("Provider charge exceeded reservation; campaign blocked")

    def uncertain(self, ident):
        with self.connect() as db:
            db.execute("UPDATE calls SET status='uncertain' WHERE id=? AND status IN ('pending','parallel_pending')", (ident,))

    def quarantine_timeout(self, ident):
        """Hold a timed-out request in full and stop its episode, with no retry.

        Call only for an observed transport TimeoutError. Other uncertainties and
        confirmed limit violations still stop the entire campaign.
        """
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM calls WHERE id=?', (ident,)).fetchone()
            if not row or row['status'] not in {'pending', 'parallel_pending'} or row['actual'] is not None:
                raise ValueError('Only a pending timed-out request can be quarantined')
            receipt = {'accounting': 'full reservation retained, actual charge unknown',
                       'error_type': 'TimeoutError', 'stop_scope': 'episode', 'created': time.time()}
            db.execute("UPDATE calls SET status='quarantined_timeout',receipt=? WHERE id=?",
                       (json.dumps(receipt), ident))

    def retain_failed_reservation(self, ident, reason):
        """Close an inspected failed request without releasing its dollar allowance.

        This is a manual audit operation, never an automatic HTTP retry. It does
        not claim a zero charge or a provider-confirmed charge. Pending requests
        and confirmed overruns cannot be acknowledged this way.
        """
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            raise ValueError('A concrete audit reason is required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM calls WHERE id=?', (ident,)).fetchone()
            if not row or row['status'] != 'uncertain' or row['actual'] is not None:
                raise ValueError('Only an uncertain failed call can retain its reservation')
            receipt = {'accounting': 'full reservation retained, actual charge unknown',
                       'reason': reason.strip(), 'reviewed_at': time.time()}
            db.execute("UPDATE calls SET status='reserved_error',receipt=? WHERE id=?",
                       (json.dumps(receipt), ident))
        return {'call_id': ident, 'retained_usd': row['reserved'], **receipt}
