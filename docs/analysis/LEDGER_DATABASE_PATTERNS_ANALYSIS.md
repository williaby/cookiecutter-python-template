# Ledger & Database Patterns Analysis for Cookiecutter Template

**Date**: 2025-11-17  
**Purpose**: Identify valuable database and ledger architecture patterns for enhancing the cookiecutter-python-template

## Executive Summary

This analysis identifies key patterns from Python ledger and financial systems that should be incorporated into the cookiecutter template to enable developers building database-heavy or ledger-based applications.

---

## 1. DATABASE PATTERNS

### 1.1 Repository Pattern with SQLAlchemy

**Key Insight**: Separate domain logic from data access layer to improve testability and maintainability.

**Implementation Approach**:
```python
# Pattern from Cosmic Python (Architecture Patterns in Python)
# - Unified interface for database operations
# - Supports multiple backends (SQLite, PostgreSQL, MySQL)
# - Enables dependency injection for testing

class RepositoryBase(Generic[T]):
    """Base repository for database operations"""
    def __init__(self, session: Session):
        self.session = session
    
    def add(self, entity: T) -> T:
        self.session.add(entity)
        return entity
    
    def get_by_id(self, id: int) -> Optional[T]:
        return self.session.query(self.model).filter(self.model.id == id).first()
    
    def list_all(self) -> List[T]:
        return self.session.query(self.model).all()
    
    def update(self, entity: T) -> T:
        self.session.merge(entity)
        return entity
    
    def delete(self, entity: T) -> None:
        self.session.delete(entity)
```

**Template Enhancement**:
- Add optional SQLAlchemy integration module (`src/{{project_slug}}/db/repository.py`)
- Provide base models with common patterns (timestamps, soft deletes, audit trails)
- Include dependency injection setup for repositories

---

### 1.2 Unit of Work Pattern

**Key Insight**: Coordinate multiple repository operations within a single transaction.

**Implementation Approach**:
```python
class UnitOfWork:
    """Coordinates multiple repositories in a single transaction"""
    def __init__(self, session: Session):
        self.session = session
        self.accounts = AccountRepository(session)
        self.transactions = TransactionRepository(session)
        self.journal_entries = JournalEntryRepository(session)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
    
    def commit(self):
        self.session.commit()
    
    def rollback(self):
        self.session.rollback()
```

**Template Enhancement**:
- Add `UnitOfWork` pattern to SQLAlchemy integration
- Include transaction context manager for atomic operations
- Document rollback and error handling

---

### 1.3 Database Migrations with Alembic

**Key Insight**: Version-controlled, reversible schema changes that integrate with CI/CD.

**Django Ledger Pattern**: Uses Django migrations extensively
**Accounting System Pattern**: SQLite with structured schema versioning

**Template Enhancement**:
- Add Alembic configuration option (like `include_database` flag)
- Generate migration scripts structure
- Integrate migration verification into CI pipeline
- Include migration rollback testing in pytest

```bash
# Pre-commit hook for migrations
poetry run alembic revision --autogenerate -m "description"
poetry run alembic upgrade head
poetry run alembic downgrade -1  # Test rollback
```

---

### 1.4 Connection Pooling & Performance

**Key Insight**: Efficient database connection management, especially for high-throughput applications.

**Configuration Patterns**:
```python
# From best practices in production systems
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,           # Number of connections to keep in pool
    max_overflow=40,        # Additional connections beyond pool_size
    pool_pre_ping=True,     # Verify connections before use
    pool_recycle=3600,      # Recycle connections after 1 hour
    echo=False              # Log SQL (set True for debugging)
)
```

**Template Enhancement**:
- Add database connection configuration patterns
- Include performance tuning guidelines in documentation
- Document connection pool sizing based on workload

---

## 2. LEDGER & DOUBLE-ENTRY BOOKKEEPING PATTERNS

### 2.1 Double-Entry Accounting Schema

**Key Insight**: Every financial transaction creates two journal entries (debit and credit) to maintain balance.

**Core Schema Components**:

```python
# Chart of Accounts (accounts table)
- account_id (PK)
- account_code (unique)
- account_name
- account_type (Asset, Liability, Equity, Income, Expense)
- normal_balance (Debit or Credit)

# Journal Entries (journal_entries table)
- entry_id (PK)
- transaction_id (FK)
- account_id (FK)
- entry_date
- description
- debit_amount (nullable)
- credit_amount (nullable)
- reference_number

# Transactions (transactions table)
- transaction_id (PK)
- transaction_date
- description
- status (posted, draft, reversed)
- created_at
- posted_at
```

**Constraints**:
```sql
-- Ensure debit OR credit, not both
CHECK (
    (debit_amount IS NOT NULL AND credit_amount IS NULL) OR
    (debit_amount IS NULL AND credit_amount IS NOT NULL)
)

-- Ensure entries balance for each transaction
-- SUM(debit) = SUM(credit) for all entries in transaction
```

**Template Enhancement**:
- Add optional `ledger` module with double-entry patterns
- Include Pydantic models for Account, JournalEntry, Transaction
- Document accounting equation: Assets = Liabilities + Equity

---

### 2.2 Ledger Posting & Validation

**Key Insight**: Transactions move through states (draft → posted → reversed) with strict validation rules.

**State Machine Pattern**:
```python
class TransactionState(Enum):
    DRAFT = "draft"           # Being created, can be edited
    POSTED = "posted"         # Locked, creates ledger entries
    REVERSED = "reversed"     # Voided but audit trail preserved

class LedgerValidator:
    @staticmethod
    def validate_entry(entry: JournalEntry) -> List[str]:
        """Validate individual journal entry"""
        errors = []
        
        # Ensure only debit OR credit
        if (entry.debit_amount and entry.credit_amount):
            errors.append("Entry cannot have both debit and credit")
        
        # Ensure amount is positive
        amount = entry.debit_amount or entry.credit_amount
        if amount <= 0:
            errors.append("Amount must be positive")
        
        # Verify account exists and is active
        if not entry.account.is_active:
            errors.append(f"Account {entry.account.code} is inactive")
        
        return errors
    
    @staticmethod
    def validate_transaction(transaction: Transaction) -> List[str]:
        """Validate complete transaction balances"""
        errors = []
        
        debits = sum(e.debit_amount or 0 for e in transaction.entries)
        credits = sum(e.credit_amount or 0 for e in transaction.entries)
        
        if abs(debits - credits) > 0.01:  # Allow for floating point
            errors.append(f"Transaction does not balance: Debits={debits}, Credits={credits}")
        
        if not transaction.entries:
            errors.append("Transaction must have at least 2 entries")
        
        return errors
```

**Template Enhancement**:
- Add `TransactionValidator` class
- Include pre/post posting hooks
- Document state transitions and business rules

---

### 2.3 Account Hierarchies & Ledger Reports

**Key Insight**: Accounts organize into hierarchies for reporting (total assets, net income, etc.).

**Hierarchical Schema**:
```python
class Account(Base):
    __tablename__ = "accounts"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(unique=True, index=True)
    name: Mapped[str]
    account_type: Mapped[AccountType]  # Asset, Liability, etc.
    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey("accounts.id"))
    
    # Relationships
    children: Mapped[List["Account"]] = relationship(back_populates="parent")
    parent: Mapped[Optional["Account"]] = relationship(back_populates="children")
    entries: Mapped[List["JournalEntry"]] = relationship()

class AccountType(str, Enum):
    ASSET = "asset"              # Normal balance: Debit
    LIABILITY = "liability"      # Normal balance: Credit
    EQUITY = "equity"            # Normal balance: Credit
    INCOME = "income"            # Normal balance: Credit
    EXPENSE = "expense"          # Normal balance: Debit
    CONTRA_ASSET = "contra_asset"  # Normal balance: Credit
```

**Report Generation**:
```python
def generate_trial_balance(session: Session, date: datetime) -> Dict[str, Decimal]:
    """Generate trial balance (all accounts with balances)"""
    query = """
    SELECT 
        a.code,
        a.name,
        COALESCE(SUM(je.debit_amount), 0) - COALESCE(SUM(je.credit_amount), 0) as balance
    FROM accounts a
    LEFT JOIN journal_entries je ON a.id = je.account_id
    WHERE je.entry_date <= :date
    GROUP BY a.id, a.code, a.name
    ORDER BY a.code
    """
    return execute_report_query(session, query, date)

def generate_income_statement(session: Session, start_date, end_date) -> Dict:
    """Generate income statement (Income - Expenses)"""
    # Query income accounts
    # Query expense accounts
    # Calculate net income
    pass
```

**Template Enhancement**:
- Add account hierarchy models
- Include report generation functions
- Document financial statement generation

---

## 3. DATA VALIDATION PATTERNS

### 3.1 Financial Data Validation

**Key Insight**: Special validation rules for financial data (amounts, currencies, date ranges).

```python
from decimal import Decimal
from pydantic import BaseModel, field_validator, model_validator

class JournalEntrySchema(BaseModel):
    account_id: int
    debit_amount: Optional[Decimal] = None
    credit_amount: Optional[Decimal] = None
    currency: str = "USD"
    entry_date: datetime
    description: str
    
    @field_validator('debit_amount', 'credit_amount', mode='before')
    @classmethod
    def validate_amount(cls, v):
        if v is None:
            return v
        
        # Convert to Decimal for precision
        if isinstance(v, str):
            v = Decimal(v)
        elif isinstance(v, float):
            v = Decimal(str(v))
        
        # Ensure positive
        if v < 0:
            raise ValueError("Amount must be positive")
        
        # Ensure reasonable precision (2 decimal places for most currencies)
        if v.as_tuple().exponent < -2:
            raise ValueError("Amount has too many decimal places")
        
        return v
    
    @model_validator(mode='after')
    def validate_debit_or_credit(self):
        has_debit = self.debit_amount is not None
        has_credit = self.credit_amount is not None
        
        # XOR: Must have exactly one of debit or credit
        if not (has_debit ^ has_credit):
            raise ValueError("Entry must have either debit OR credit, not both or neither")
        
        return self

class TransactionSchema(BaseModel):
    entries: List[JournalEntrySchema]
    transaction_date: datetime
    
    @model_validator(mode='after')
    def validate_transaction_balance(self):
        debits = sum(e.debit_amount or Decimal(0) for e in self.entries)
        credits = sum(e.credit_amount or Decimal(0) for e in self.entries)
        
        if abs(debits - credits) > Decimal('0.01'):
            raise ValueError(
                f"Transaction does not balance: Debits={debits}, Credits={credits}"
            )
        
        if len(self.entries) < 2:
            raise ValueError("Transaction must have at least 2 journal entries")
        
        return self
```

**Template Enhancement**:
- Add financial validation schemas to Pydantic models
- Include amount validation with Decimal for precision
- Document validation rules and error messages

---

### 3.2 Idempotency & Duplicate Prevention

**Key Insight**: Prevent duplicate transactions and ensure safe retries.

```python
class TransactionIdempotencyKey(Base):
    """Prevent duplicate transactions via idempotency key"""
    __tablename__ = "transaction_idempotency_keys"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(unique=True, index=True)
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

class LedgerService:
    def create_transaction_idempotent(
        self,
        idempotency_key: str,
        transaction_data: TransactionSchema
    ) -> Transaction:
        """Create transaction with idempotency guarantee"""
        # Check if already exists
        existing = self.session.query(TransactionIdempotencyKey).filter(
            TransactionIdempotencyKey.idempotency_key == idempotency_key
        ).first()
        
        if existing:
            return self.session.query(Transaction).get(existing.transaction_id)
        
        # Create new transaction
        transaction = Transaction(**transaction_data.dict())
        self.session.add(transaction)
        self.session.flush()
        
        # Store idempotency key
        key_record = TransactionIdempotencyKey(
            idempotency_key=idempotency_key,
            transaction_id=transaction.id
        )
        self.session.add(key_record)
        self.session.commit()
        
        return transaction
```

**Template Enhancement**:
- Add idempotency key table and logic
- Document safe retry patterns
- Include integration tests for idempotency

---

## 4. TESTING PATTERNS

### 4.1 Database Testing with Pytest

**Key Insight**: Efficient database testing with transactions and factories.

```python
# conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db_session():
    """Provide in-memory SQLite session for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()
    engine.dispose()

@pytest.fixture
def sample_accounts(db_session):
    """Create sample chart of accounts"""
    accounts = [
        Account(code="1000", name="Cash", account_type=AccountType.ASSET),
        Account(code="2000", name="Accounts Payable", account_type=AccountType.LIABILITY),
        Account(code="3000", name="Capital Stock", account_type=AccountType.EQUITY),
        Account(code="4000", name="Sales Revenue", account_type=AccountType.INCOME),
        Account(code="5000", name="Operating Expenses", account_type=AccountType.EXPENSE),
    ]
    db_session.add_all(accounts)
    db_session.commit()
    return accounts

# tests/unit/test_ledger.py
def test_transaction_must_balance(db_session, sample_accounts):
    """Verify transaction validation enforces balance rule"""
    transaction = Transaction(transaction_date=datetime.now())
    
    # Add unbalanced entries
    entry1 = JournalEntry(
        transaction=transaction,
        account=sample_accounts[0],
        debit_amount=Decimal("100.00")
    )
    entry2 = JournalEntry(
        transaction=transaction,
        account=sample_accounts[1],
        credit_amount=Decimal("50.00")  # Imbalanced!
    )
    
    db_session.add(transaction)
    
    # Validation should fail
    with pytest.raises(ValidationError):
        validator = LedgerValidator()
        errors = validator.validate_transaction(transaction)
        assert errors  # Should have balance error

def test_double_entry_integrity(db_session, sample_accounts):
    """Verify double-entry bookkeeping enforces two entries minimum"""
    transaction = Transaction(transaction_date=datetime.now())
    
    # Add only one entry (should fail)
    entry = JournalEntry(
        transaction=transaction,
        account=sample_accounts[0],
        debit_amount=Decimal("100.00")
    )
    
    db_session.add(transaction)
    db_session.flush()
    
    # Should fail validation
    validator = LedgerValidator()
    errors = validator.validate_transaction(transaction)
    assert "at least 2 entries" in errors[0]

def test_account_normal_balance(db_session, sample_accounts):
    """Verify account balance respects normal balance rule"""
    asset = sample_accounts[0]  # Assets normally debit
    
    transaction = Transaction(transaction_date=datetime.now())
    entry = JournalEntry(
        transaction=transaction,
        account=asset,
        debit_amount=Decimal("100.00")  # Should be debit for asset
    )
    
    db_session.add(transaction)
    db_session.commit()
    
    balance = calculate_account_balance(db_session, asset.id)
    assert balance > 0  # Asset should show positive balance
```

**Template Enhancement**:
- Add ledger-specific pytest fixtures
- Include sample chart of accounts
- Document transaction factory patterns
- Add integration tests for state transitions

---

### 4.2 Ledger Report Testing

**Key Insight**: Test report generation with known data sets.

```python
def test_trial_balance_equals_zero(db_session, sample_accounts):
    """Trial balance debit and credit totals must be equal"""
    # Create balanced transactions
    transaction = create_balanced_transaction(
        db_session,
        sample_accounts[0],  # Cash
        sample_accounts[1],  # A/P
        Decimal("1000.00")
    )
    db_session.commit()
    
    report = generate_trial_balance(db_session, datetime.now())
    
    total_debits = sum(v for v in report.values() if v > 0)
    total_credits = sum(abs(v) for v in report.values() if v < 0)
    
    assert abs(total_debits - total_credits) < Decimal('0.01')

def test_income_statement_net_income(db_session):
    """Income statement net income = revenue - expenses"""
    # Setup accounts
    revenue = Account(code="4000", name="Revenue", account_type=AccountType.INCOME)
    expense = Account(code="5000", name="Expense", account_type=AccountType.EXPENSE)
    db_session.add_all([revenue, expense])
    db_session.commit()
    
    # Create transactions: $1000 revenue, $300 expense
    t1 = create_balanced_transaction(db_session, revenue, 
                                     sample_accounts[0], Decimal("1000.00"))
    t2 = create_balanced_transaction(db_session, expense,
                                     sample_accounts[0], Decimal("300.00"))
    db_session.commit()
    
    statement = generate_income_statement(
        db_session,
        datetime(2024, 1, 1),
        datetime(2024, 1, 31)
    )
    
    assert statement['net_income'] == Decimal("700.00")
```

---

## 5. CONFIGURATION PATTERNS

### 5.1 Database Configuration Management

**Key Insight**: Environment-based configuration with multiple database options.

```python
# src/{{project_slug}}/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class DatabaseSettings(BaseSettings):
    """Database configuration from environment"""
    
    # Connection string
    db_url: str = "sqlite:///./app.db"
    
    # Pool settings
    pool_size: int = 20
    max_overflow: int = 40
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    
    # Query logging
    echo_sql: bool = False
    
    # Migrations
    auto_migrate: bool = False
    migration_path: str = "alembic"
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")

class LedgerSettings(BaseSettings):
    """Ledger-specific configuration"""
    
    # Rounding
    decimal_places: int = 2
    
    # Transaction posting
    allow_future_dated_transactions: bool = False
    
    # Auditing
    track_modifications: bool = True
    audit_table_prefix: str = "audit_"
    
    model_config = SettingsConfigDict(env_prefix="LEDGER_")

class Settings(BaseSettings):
    """Combined application settings"""
    
    database: DatabaseSettings = DatabaseSettings()
    ledger: LedgerSettings = LedgerSettings()
    
    model_config = SettingsConfigDict(env_file=".env")

# Usage
settings = Settings()
engine = create_engine(
    settings.database.db_url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    echo=settings.database.echo_sql
)
```

**Template Enhancement**:
- Add `DatabaseSettings` to config module
- Include optional `LedgerSettings`
- Document environment variable mapping

---

### 5.2 Ledger Initialization Configuration

**Key Insight**: Configurable chart of accounts and ledger rules.

```yaml
# config/chart_of_accounts.yaml
accounts:
  assets:
    - code: "1000"
      name: "Cash"
      account_type: "asset"
      subaccounts:
        - code: "1010"
          name: "Checking Account"
        - code: "1020"
          name: "Savings Account"
  
  liabilities:
    - code: "2000"
      name: "Accounts Payable"
      account_type: "liability"
  
  equity:
    - code: "3000"
      name: "Capital Stock"
      account_type: "equity"
  
  income:
    - code: "4000"
      name: "Sales Revenue"
      account_type: "income"
  
  expenses:
    - code: "5000"
      name: "Operating Expenses"
      account_type: "expense"
      subaccounts:
        - code: "5100"
          name: "Salaries"
        - code: "5200"
          name: "Rent"

ledger_rules:
  decimal_places: 2
  require_approval_for_reversals: true
  allow_negative_balances:
    accounts:
      - "2000"  # A/P can be negative
  reporting:
    fiscal_year_start: "01-01"
    reporting_currencies:
      - "USD"
      - "EUR"
```

**Template Enhancement**:
- Add YAML configuration for chart of accounts
- Include loader function to initialize ledger
- Document account hierarchy options

---

## 6. DOCUMENTATION PATTERNS

### 6.1 Database Schema Documentation

**Key Insight**: Schema documentation with relationships and constraints.

```markdown
# Database Schema

## accounts
Chart of accounts for double-entry bookkeeping.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Account identifier |
| code | VARCHAR(20) | UNIQUE, NOT NULL | Account code (e.g., "1000") |
| name | VARCHAR(255) | NOT NULL | Account name |
| account_type | ENUM | NOT NULL | Asset, Liability, Equity, Income, Expense |
| parent_id | INTEGER | FK → accounts.id | Parent account for hierarchy |
| is_active | BOOLEAN | DEFAULT TRUE | Soft delete flag |
| created_at | TIMESTAMP | NOT NULL | Creation timestamp |

**Indexes**: 
- code (unique)
- parent_id (for hierarchy queries)
- account_type (for statement generation)

## journal_entries
Individual debit/credit entries for transactions.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Entry identifier |
| transaction_id | INTEGER | FK → transactions.id, NOT NULL | Associated transaction |
| account_id | INTEGER | FK → accounts.id, NOT NULL | Account posting to |
| debit_amount | DECIMAL(15,2) | Can be NULL | Debit amount (only one of debit/credit) |
| credit_amount | DECIMAL(15,2) | Can be NULL | Credit amount (only one of debit/credit) |
| entry_date | DATE | NOT NULL | Entry date |
| description | VARCHAR(500) | | Entry description |

**Constraints**:
```sql
CHECK ((debit_amount IS NOT NULL AND credit_amount IS NULL) OR 
       (debit_amount IS NULL AND credit_amount IS NOT NULL))
```

**Indexes**:
- transaction_id
- account_id
- entry_date (for period queries)
```

**Template Enhancement**:
- Add database schema documentation template
- Include ER diagram examples (PlantUML)
- Document constraint rationales

---

### 6.2 Ledger Architecture Diagrams

**Key Insight**: Visual documentation of double-entry flow.

```
Transaction Lifecycle:
┌─────────────┐
│   DRAFT     │  (Editable)
└──────┬──────┘
       │ post()
       ↓
┌─────────────┐
│   POSTED    │  (Ledger entries created)
└──────┬──────┘
       │ reverse()
       ↓
┌─────────────┐
│  REVERSED   │  (Audit trail preserved)
└─────────────┘

Double-Entry Posting:
┌──────────────────────────────────────┐
│        Cash Sale for $100            │
├──────────────────────────────────────┤
│  Debit: Cash           $100          │
│  Credit: Sales Revenue          $100 │
└──────────────────────────────────────┘
      ↓
┌──────────────────────────────────────┐
│   Journal Entry Posted               │
├──────────────────────────────────────┤
│ Account 1000 (Asset): +$100 (debit)  │
│ Account 4000 (Income): -$100 (credit)│
└──────────────────────────────────────┘
```

**Template Enhancement**:
- Add PlantUML templates for transaction flow
- Include ledger posting sequence diagrams
- Document report generation flow

---

## 7. CI/CD PATTERNS FOR DATABASES

### 7.1 Database Migration Testing in CI

**Key Insight**: Validate migrations run cleanly in CI without manual fixes.

```yaml
# .github/workflows/ci.yml (additions for database projects)

jobs:
  test-migrations:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test_db
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "poetry"
      
      - name: Install dependencies
        run: poetry install --with dev
      
      - name: Run migrations (upgrade)
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test_db
        run: poetry run alembic upgrade head
      
      - name: Verify schema
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test_db
        run: poetry run python scripts/verify_schema.py
      
      - name: Test migration rollback
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test_db
        run: |
          poetry run alembic downgrade -1
          poetry run alembic upgrade head
      
      - name: Run database tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test_db
        run: poetry run pytest tests/integration/test_database.py -v
```

**Template Enhancement**:
- Add PostgreSQL service for CI
- Include migration upgrade/downgrade tests
- Add schema verification script

---

### 7.2 Ledger Integrity Checks in CI

**Key Insight**: Automated tests for ledger consistency.

```python
# scripts/verify_ledger_integrity.py
"""Verify ledger integrity constraints"""

def verify_trial_balance(session: Session) -> bool:
    """Verify all transactions are balanced"""
    query = """
    SELECT t.id, 
           COALESCE(SUM(je.debit_amount), 0) as debits,
           COALESCE(SUM(je.credit_amount), 0) as credits
    FROM transactions t
    LEFT JOIN journal_entries je ON t.id = je.transaction_id
    WHERE t.status = 'posted'
    GROUP BY t.id
    HAVING COALESCE(SUM(je.debit_amount), 0) != COALESCE(SUM(je.credit_amount), 0)
    """
    
    unbalanced = session.execute(text(query)).fetchall()
    if unbalanced:
        print(f"FAIL: {len(unbalanced)} unbalanced transactions found")
        for row in unbalanced:
            print(f"  Transaction {row.id}: Debits={row.debits}, Credits={row.credits}")
        return False
    
    print("PASS: All transactions balanced")
    return True

def verify_account_codes_unique(session: Session) -> bool:
    """Verify no duplicate account codes"""
    duplicates = session.query(Account.code, func.count(Account.id))\
        .group_by(Account.code)\
        .having(func.count(Account.id) > 1)\
        .all()
    
    if duplicates:
        print(f"FAIL: {len(duplicates)} duplicate account codes found")
        return False
    
    print("PASS: All account codes unique")
    return True
```

---

## 8. PERFORMANCE PATTERNS

### 8.1 Ledger Query Optimization

**Key Insight**: Efficient queries for common ledger operations.

```python
# Instead of:
def slow_account_balance(session, account_id, as_of_date):
    entries = session.query(JournalEntry).filter(
        JournalEntry.account_id == account_id,
        JournalEntry.entry_date <= as_of_date
    ).all()
    
    balance = Decimal(0)
    for entry in entries:
        if entry.debit_amount:
            balance += entry.debit_amount
        if entry.credit_amount:
            balance -= entry.credit_amount
    
    return balance

# Use optimized query:
def fast_account_balance(session, account_id, as_of_date):
    query = """
    SELECT COALESCE(SUM(COALESCE(debit_amount, 0) - 
                        COALESCE(credit_amount, 0)), 0) as balance
    FROM journal_entries
    WHERE account_id = :account_id
      AND entry_date <= :as_of_date
    """
    
    result = session.execute(
        text(query),
        {"account_id": account_id, "as_of_date": as_of_date}
    ).scalar()
    
    return Decimal(result)
```

**Indexing Strategy**:
```sql
-- Account balance queries
CREATE INDEX idx_journal_entries_account_date 
  ON journal_entries(account_id, entry_date);

-- Transaction queries
CREATE INDEX idx_journal_entries_transaction 
  ON journal_entries(transaction_id);

-- Date range queries
CREATE INDEX idx_transactions_date 
  ON transactions(transaction_date);

-- Report generation
CREATE INDEX idx_accounts_type 
  ON accounts(account_type);
```

**Template Enhancement**:
- Add optimized query patterns
- Include indexing strategies
- Document query performance benchmarks

---

### 8.2 Materialized Views for Reports

**Key Insight**: Pre-computed summaries for fast report generation.

```python
# Create materialized view of account balances
class AccountBalance(Base):
    """Materialized view of account balances (refreshed periodically)"""
    __tablename__ = "account_balances_mv"
    
    account_id: Mapped[int] = mapped_column(primary_key=True)
    balance_date: Mapped[date] = mapped_column(primary_key=True)
    balance: Mapped[Decimal]
    last_refreshed: Mapped[datetime]

def refresh_account_balance_view(session: Session, target_date: date = None):
    """Refresh materialized view"""
    if target_date is None:
        target_date = date.today()
    
    # Clear old data
    session.execute(
        delete(AccountBalance).where(AccountBalance.balance_date == target_date)
    )
    
    # Recalculate balances
    query = """
    INSERT INTO account_balances_mv (account_id, balance_date, balance, last_refreshed)
    SELECT 
        a.id,
        :balance_date,
        COALESCE(SUM(COALESCE(je.debit_amount, 0) - 
                     COALESCE(je.credit_amount, 0)), 0),
        NOW()
    FROM accounts a
    LEFT JOIN journal_entries je ON a.id = je.account_id
    WHERE je.entry_date <= :balance_date
    GROUP BY a.id
    """
    
    session.execute(text(query), {"balance_date": target_date})
    session.commit()
```

---

## 9. RECOMMENDED COOKIECUTTER ENHANCEMENTS

### 9.1 New Optional Module: `include_database`

```json
{
  "include_database": {
    "type": "string",
    "enum": ["none", "sqlalchemy", "sqlalchemy-with-migrations", "sqlalchemy-ledger"],
    "default": "none",
    "description": "Add database support with SQLAlchemy"
  }
}
```

When selected, generates:
- `src/{{project_slug}}/db/base.py` - Base models and session management
- `src/{{project_slug}}/db/models.py` - Domain models
- `src/{{project_slug}}/db/repository.py` - Repository pattern base classes
- `src/{{project_slug}}/db/unit_of_work.py` - Unit of Work pattern
- Migration templates (if migrations selected)
- Database configuration in `core/config.py`

### 9.2 Ledger Module Option

When `include_database == "sqlalchemy-ledger"`:
- `src/{{project_slug}}/ledger/models.py` - Account, JournalEntry, Transaction
- `src/{{project_slug}}/ledger/validators.py` - Double-entry validation
- `src/{{project_slug}}/ledger/services.py` - Posting, reversal logic
- `src/{{project_slug}}/ledger/reports.py` - Trial balance, income statement
- `tests/fixtures/ledger_fixtures.py` - Sample chart of accounts
- `docs/guides/ledger-architecture.md` - Ledger pattern documentation

### 9.3 Pre-commit Hook for Migrations

```yaml
- repo: local
  hooks:
    - id: check-migrations
      name: Check database migrations
      entry: poetry run alembic current
      language: system
      stages: [commit]
      pass_filenames: false
```

### 9.4 CI/CD Templates

Add to workflows:
- Database service (PostgreSQL option)
- Migration testing jobs
- Ledger integrity checks
- Performance benchmarks

### 9.5 Documentation Templates

Add to `docs/guides/`:
- `database-architecture.md` - Repository, Unit of Work patterns
- `ledger-patterns.md` - Double-entry bookkeeping guide
- `migration-strategy.md` - Alembic usage and best practices
- `ledger-api.md` - Posting, reversal, reporting APIs

---

## 10. IMPLEMENTATION PRIORITY

| Priority | Enhancement | Effort | Impact |
|----------|-------------|--------|--------|
| P0 | SQLAlchemy base models + session management | 2h | High - enables all DB projects |
| P0 | Repository pattern base class | 1h | High - core abstraction |
| P1 | Database configuration in Settings | 1h | Medium - needed for flexibility |
| P1 | Pydantic schemas for common patterns | 2h | Medium - data validation |
| P2 | Alembic migration scaffolding | 2h | Medium - for production DB projects |
| P2 | Ledger module (optional) | 6h | Low - specialized use case |
| P3 | Materialized view examples | 2h | Low - optimization patterns |
| P3 | Report generation templates | 3h | Low - specialized use case |

---

## Conclusion

The ledger and database patterns identified provide valuable templates for projects requiring:

1. **Multi-user database applications** - Repository, Unit of Work patterns
2. **Financial/accounting systems** - Double-entry validation, state machines
3. **High-throughput applications** - Connection pooling, materialized views
4. **Data integrity critical systems** - Validation patterns, idempotency keys

Implementing the P0/P1 enhancements would make the cookiecutter template suitable for ~60% more Python projects while maintaining simplicity for basic use cases.

