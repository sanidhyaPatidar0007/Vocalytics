"""Create a demo organization with realistic sample data (optional, for evaluation).

It creates "Demo Retail Co" (Premium plan) with one Organization Admin and one
Data Analyst, then imports three synthetic CSV datasets that contain the kinds
of problems the app is built to find: missing values, exact duplicates,
inconsistent category spellings, stray whitespace, an outlier and
identifiers with leading zeros. Passwords are generated randomly and printed
once; nothing is hardcoded. Every action is performed as an existing Super
Admin, so it appears in the audit log.

Usage (PowerShell, from the project root, after create_super_admin.py):

    python scripts\\seed_demo_data.py
"""

from __future__ import annotations

import io
import secrets
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import select  # noqa: E402

from auth.authorization import load_context  # noqa: E402
from core.constants import Permission, Role  # noqa: E402
from core.errors import AppError  # noqa: E402
from core.logging_setup import configure_logging  # noqa: E402
from database import init_database, session_scope  # noqa: E402
from database.models import Organization, User  # noqa: E402
from modules import organization_admin, super_admin  # noqa: E402
from services import dataset_service  # noqa: E402

DEMO_ORG = "Demo Retail Co"
PRODUCTS = {
    "Widget": ("Hardware", 24.50),
    "Gadget": ("Hardware", 49.99),
    "Gizmo": ("Electronics", 129.00),
    "Doohickey": ("Electronics", 79.25),
    "Sprocket": ("Parts", 5.75),
    "Flange": ("Parts", 12.40),
}
REGIONS = ["North", "South", "East", "West"]


def generate_password() -> str:
    """Random password that satisfies the policy (all four character classes)."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_"
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(16))
        if (
            any(c.islower() for c in candidate)
            and any(c.isupper() for c in candidate)
            and any(c.isdigit() for c in candidate)
            and any(c in "!@#$%^&*-_" for c in candidate)
        ):
            return candidate


def sales_frame(rng: np.random.Generator, *, start: str, periods: int, first_order: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    rows = []
    order_id = first_order
    for day in dates:
        for _ in range(int(rng.integers(2, 5))):
            product = str(rng.choice(list(PRODUCTS)))
            category, price = PRODUCTS[product]
            quantity = int(rng.integers(1, 9))
            revenue = round(quantity * price, 2)
            rows.append(
                {
                    "order_id": str(order_id),
                    "order_date": day.strftime("%Y-%m-%d"),
                    "region": str(rng.choice(REGIONS)),
                    "product": product,
                    "category": category,
                    "customer_id": f"C{int(rng.integers(1, 400)):05d}",
                    "quantity": str(quantity),
                    "unit_price": f"{price:.2f}",
                    "revenue": f"{revenue:.2f}",
                    "profit": f"{revenue * float(rng.uniform(0.12, 0.35)):.2f}",
                }
            )
            order_id += 1
    return pd.DataFrame(rows)


def with_quality_problems(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    data = frame.copy()
    n = len(data)
    data.loc[rng.choice(n, size=max(1, n // 40), replace=False), "quantity"] = ""
    data.loc[rng.choice(n, size=max(1, n // 60), replace=False), "region"] = "N/A"
    data.loc[rng.choice(n, size=max(1, n // 50), replace=False), "region"] = "north"
    data.loc[rng.choice(n, size=max(1, n // 50), replace=False), "product"] = " Widget "
    data.loc[int(rng.integers(0, n)), "revenue"] = "98500.00"  # a suspicious outlier
    duplicates = data.sample(n=max(1, n // 50), random_state=7)
    return pd.concat([data, duplicates], ignore_index=True)


def customers_frame(rng: np.random.Generator) -> pd.DataFrame:
    cities = ["Pune", "Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]
    segments = ["Retail", "Wholesale", "Online"]
    rows = [
        {
            "customer_id": f"C{i:05d}",
            "customer_name": f"Customer {i}",
            "city": str(rng.choice(cities)),
            "segment": str(rng.choice(segments)),
            "signup_date": (
                pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
            ).strftime("%Y-%m-%d"),
        }
        for i in range(1, 400)
    ]
    return pd.DataFrame(rows)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    configure_logging()
    init_database(seed=True)
    with session_scope() as db:
        admin_id = (
            db.execute(select(User.id).where(User.role == Role.SUPER_ADMIN.value, User.is_active.is_(True)))
            .scalars()
            .first()
        )
        if admin_id is None:
            print("Create a Super Admin first: python scripts\\create_super_admin.py")
            return 1
        if db.execute(select(Organization.id).where(Organization.name == DEMO_ORG)).first():
            print(f"'{DEMO_ORG}' already exists; nothing was changed.")
            return 0
        platform_ctx = load_context(db, admin_id)

    rng = np.random.default_rng(42)
    admin_password, analyst_password = generate_password(), generate_password()
    try:
        org = super_admin.create_organization(
            platform_ctx,
            name=DEMO_ORG,
            contact_email="owner@demo-retail.example",
            plan_code="premium",
            admin_username="demo.admin",
            admin_email="admin@demo-retail.example",
            admin_password=admin_password,
            admin_full_name="Demo Admin",
        )
        with session_scope() as db:
            admin_user = db.execute(select(User.id).where(User.username == "demo.admin")).scalar_one()
            org_ctx = load_context(db, admin_user)
        analyst = organization_admin.create_analyst(
            org_ctx,
            username="demo.analyst",
            email="analyst@demo-retail.example",
            full_name="Demo Analyst",
            password=analyst_password,
            permissions={
                Permission.EXPORT_DATA.value,
                Permission.USE_VOICE.value,
                Permission.SAVE_REPORTS.value,
                Permission.CLEAN_DATA.value,
            },
        )
        datasets = {
            "Sales 2024": with_quality_problems(
                sales_frame(rng, start="2024-01-01", periods=366, first_order=100001), rng
            ),
            "Sales 2025 Q1": sales_frame(rng, start="2025-01-01", periods=90, first_order=200001),
            "Customers": customers_frame(rng),
        }
        for name, frame in datasets.items():
            result = dataset_service.import_dataset(
                org_ctx,
                file_name=f"{name.lower().replace(' ', '_')}.csv",
                data=_csv_bytes(frame),
                dataset_name=name,
            )
            organization_admin.grant_dataset_access(org_ctx, result["dataset_id"], analyst["id"])
            print(f"Imported '{name}': {result['rows']} rows, quality {result['quality_score']}")
    except AppError as exc:
        print(f"Seeding stopped: {exc.user_message}")
        return 1
    print(f"\nOrganization '{org['name']}' is ready (Premium plan). Sign in with:")
    print(f"  Organization Admin  demo.admin    {admin_password}")
    print(f"  Data Analyst        demo.analyst  {analyst_password}")
    print("These passwords are shown only once. Change them on the Account page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
