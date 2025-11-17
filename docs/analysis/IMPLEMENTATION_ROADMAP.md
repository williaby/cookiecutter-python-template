# Implementation Roadmap: Database & Ledger Patterns for Cookiecutter

## Executive Summary

This roadmap outlines how to enhance the `cookiecutter-python-template` with database and ledger patterns. The implementation is divided into 3 phases, allowing incremental adoption.

---

## Current State Analysis

### What the Template Already Has (✓)

1. ✓ Professional project structure
2. ✓ Poetry dependency management
3. ✓ Pytest testing framework
4. ✓ Pydantic v2 models
5. ✓ Comprehensive CI/CD workflows
6. ✓ Security scanning (Bandit, Safety, OSV)
7. ✓ Configuration system (`core/config.py`)
8. ✓ Structured logging

### What's Missing (✗)

1. ✗ Database integration (SQLAlchemy)
2. ✗ Repository pattern implementation
3. ✗ Unit of Work pattern
4. ✗ Migration support (Alembic)
5. ✗ Ledger/financial system patterns
6. ✗ Database fixtures for testing
7. ✗ Database configuration samples
8. ✗ Performance optimization patterns

---

## Phase 1: Foundation (P0) - 2-3 Days

### Goal
Provide SQLAlchemy foundation for all database projects

### Deliverables

#### 1.1 Update `cookiecutter.json`

```json
{
  "project_name": "My Project",
  ...
  "include_database": {
    "type": "string",
    "enum": ["none", "sqlalchemy", "sqlalchemy_migrations", "sqlalchemy_ledger"],
    "default": "none",
    "description": "Database support (none/sqlalchemy/sqlalchemy_migrations/sqlalchemy_ledger)"
  }
}
```

#### 1.2 Create `src/{{cookiecutter.project_slug}}/db/base.py`

```python
"""Database session and base model configuration."""

from typing import Any
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Base class for all ORM models
Base = declarative_base()

class Database:
    """Database connection and session management."""
    
    def __init__(self, url: str, echo: bool = False, **engine_kwargs: Any):
        """Initialize database connection.
        
        Args:
            url: Database connection URL
            echo: Echo SQL statements (for debugging)
            **engine_kwargs: Additional SQLAlchemy engine options
        """
        self.engine = create_engine(url, echo=echo, **engine_kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
    
    def create_tables(self) -> None:
        """Create all tables defined in models."""
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self) -> None:
        """Drop all tables (for testing)."""
        Base.metadata.drop_all(self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()


def get_db() -> Database:
    """Factory function for dependency injection."""
    from {{cookiecutter.project_slug}}.core.config import settings
    
    return Database(
        url=settings.database.db_url,
        echo=settings.database.echo_sql,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_pre_ping=settings.database.pool_pre_ping,
    )
```

#### 1.3 Create `src/{{cookiecutter.project_slug}}/db/models.py`

```python
"""Base models with common patterns."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, DateTime, Boolean, String, Index
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class SoftDeleteMixin:
    """Mixin for soft deletes (logical deletion)."""
    
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    def soft_delete(self) -> None:
        """Soft delete this record."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()


class BaseModel(Base, TimestampMixin):
    """Base model with timestamps for all entities."""
    
    __abstract__ = True
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)


class AuditableMixin:
    """Mixin for audit trail (who created/modified)."""
    
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    updated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
```

#### 1.4 Create `src/{{cookiecutter.project_slug}}/db/repository.py`

```python
"""Repository pattern implementation for clean data access layer."""

from typing import Generic, TypeVar, List, Optional
from sqlalchemy.orm import Session

T = TypeVar('T')


class RepositoryBase(Generic[T]):
    """Base repository class for all entities.
    
    Provides a clean abstraction over database operations,
    separating domain logic from data access.
    """
    
    def __init__(self, session: Session, model: type[T]):
        """Initialize repository.
        
        Args:
            session: SQLAlchemy session
            model: The ORM model class managed by this repository
        """
        self.session = session
        self.model = model
    
    def add(self, entity: T) -> T:
        """Add entity to repository.
        
        Args:
            entity: Entity instance to add
            
        Returns:
            The added entity
        """
        self.session.add(entity)
        return entity
    
    def get_by_id(self, id: int) -> Optional[T]:
        """Get entity by primary key.
        
        Args:
            id: Primary key value
            
        Returns:
            Entity if found, None otherwise
        """
        return self.session.query(self.model).filter(self.model.id == id).first()
    
    def list_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """List all entities with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum records to return
            
        Returns:
            List of entities
        """
        return self.session.query(self.model).offset(skip).limit(limit).all()
    
    def count(self) -> int:
        """Count total entities.
        
        Returns:
            Number of entities
        """
        return self.session.query(self.model).count()
    
    def update(self, entity: T) -> T:
        """Update entity.
        
        Args:
            entity: Entity with updated values
            
        Returns:
            Updated entity
        """
        self.session.merge(entity)
        return entity
    
    def delete(self, entity: T) -> None:
        """Delete entity.
        
        Args:
            entity: Entity to delete
        """
        self.session.delete(entity)
    
    def delete_by_id(self, id: int) -> None:
        """Delete entity by primary key.
        
        Args:
            id: Primary key value
        """
        entity = self.get_by_id(id)
        if entity:
            self.delete(entity)
```

#### 1.5 Create `src/{{cookiecutter.project_slug}}/db/unit_of_work.py`

```python
"""Unit of Work pattern for coordinating multiple repositories."""

from typing import Dict, Type
from sqlalchemy.orm import Session
from .repository import RepositoryBase


class UnitOfWork:
    """Coordinates multiple repositories within a single transaction.
    
    Provides ACID guarantees for complex operations involving multiple entities.
    """
    
    def __init__(self, session: Session):
        """Initialize Unit of Work.
        
        Args:
            session: SQLAlchemy session for this unit of work
        """
        self.session = session
        self._repositories: Dict[Type, RepositoryBase] = {}
    
    def get_repository(self, model_class: Type[T]) -> RepositoryBase[T]:
        """Get or create a repository for a model class.
        
        Args:
            model_class: The ORM model class
            
        Returns:
            Repository instance for the model
        """
        if model_class not in self._repositories:
            self._repositories[model_class] = RepositoryBase(
                self.session, model_class
            )
        return self._repositories[model_class]
    
    def commit(self) -> None:
        """Commit all changes in this unit of work."""
        self.session.commit()
    
    def rollback(self) -> None:
        """Rollback all changes in this unit of work."""
        self.session.rollback()
    
    def flush(self) -> None:
        """Flush pending changes without committing."""
        self.session.flush()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic rollback on error."""
        if exc_type is not None:
            self.rollback()
        else:
            try:
                self.commit()
            except Exception:
                self.rollback()
                raise
        return False
```

#### 1.6 Update `src/{{cookiecutter.project_slug}}/core/config.py`

Add `DatabaseSettings` class:

```python
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    """Database configuration from environment variables."""
    
    # Connection
    db_url: str = "sqlite:///./app.db"
    
    # Connection pooling
    pool_size: int = 20
    max_overflow: int = 40
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    
    # Query logging
    echo_sql: bool = False
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class Settings(BaseSettings):
    """Application settings."""
    
    # ... existing settings ...
    database: DatabaseSettings = DatabaseSettings()
```

#### 1.7 Update `pyproject.toml`

Add SQLAlchemy to dependencies:

```toml
[project.optional-dependencies]
database = [
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
]

[project.optional-dependencies]
dev = [
    # ... existing ...
    "factory-boy>=3.3.0",
    "pytest-postgresql>=5.0.0",
]
```

#### 1.8 Create `tests/conftest.py` with Database Fixtures

```python
"""Test configuration and fixtures."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from {{cookiecutter.project_slug}}.db.base import Base


@pytest.fixture(scope="function")
def db_engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a new database session for a test."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()
```

---

## Phase 2: Advanced Patterns (P1) - 3-4 Days

### Goal
Add Alembic migrations and advanced validation patterns

### Deliverables

#### 2.1 Alembic Migration Setup

**When `include_database` includes `"_migrations"`:**

- Generate `alembic/` directory with:
  - `env.py` - Migration environment configuration
  - `script.py.mako` - Migration template
  - `versions/` - Migration scripts directory
  - `alembic.ini` - Configuration file

#### 2.2 Financial Data Validation (`pydantic_models.py`)

```python
"""Pydantic schemas with financial validation."""

from decimal import Decimal
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, field_validator, model_validator


class MoneyField(BaseModel):
    """Validated monetary amount."""
    amount: Decimal
    currency: str = "USD"
    
    @field_validator('amount', mode='before')
    @classmethod
    def validate_amount(cls, v):
        """Validate monetary amount."""
        if v is None:
            raise ValueError("Amount cannot be None")
        
        # Convert to Decimal
        if isinstance(v, str):
            v = Decimal(v)
        elif isinstance(v, float):
            v = Decimal(str(v))
        elif not isinstance(v, Decimal):
            v = Decimal(v)
        
        # Check positive
        if v < 0:
            raise ValueError("Amount must be positive")
        
        # Check precision (max 2 decimal places for most currencies)
        if v.as_tuple().exponent < -2:
            raise ValueError(f"Amount has too many decimal places: {v}")
        
        return v


class JournalEntrySchema(BaseModel):
    """Single debit or credit entry."""
    account_id: int
    debit_amount: Optional[Decimal] = None
    credit_amount: Optional[Decimal] = None
    description: str
    entry_date: datetime
    
    @field_validator('debit_amount', 'credit_amount', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert monetary amounts to Decimal."""
        if v is None:
            return v
        
        if isinstance(v, str):
            return Decimal(v)
        if isinstance(v, float):
            return Decimal(str(v))
        return Decimal(v) if not isinstance(v, Decimal) else v
    
    @model_validator(mode='after')
    def debit_xor_credit(self):
        """Ensure exactly one of debit or credit is set."""
        has_debit = self.debit_amount is not None
        has_credit = self.credit_amount is not None
        
        # XOR: exactly one must be true
        if not (has_debit ^ has_credit):
            raise ValueError(
                "Entry must have either debit_amount OR credit_amount, not both or neither"
            )
        
        return self


class TransactionSchema(BaseModel):
    """Complete transaction with multiple entries."""
    entries: list[JournalEntrySchema]
    transaction_date: datetime
    description: str
    
    @model_validator(mode='after')
    def validate_transaction_balance(self):
        """Ensure debits equal credits."""
        debits = sum(
            (e.debit_amount or Decimal(0)) for e in self.entries
        )
        credits = sum(
            (e.credit_amount or Decimal(0)) for e in self.entries
        )
        
        # Allow for floating point errors
        if abs(debits - credits) > Decimal('0.01'):
            raise ValueError(
                f"Transaction does not balance: "
                f"Debits={debits}, Credits={credits}"
            )
        
        if len(self.entries) < 2:
            raise ValueError("Transaction must have at least 2 entries")
        
        return self
```

#### 2.3 Idempotency Key Pattern

```python
"""Idempotency support for safe retries."""

from uuid import uuid4
from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .models import BaseModel


class IdempotencyMixin:
    """Add idempotency key to any entity."""
    
    idempotency_key: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )


class IdempotencyKeyRecord(BaseModel):
    """Track idempotency keys for duplicate prevention."""
    
    __tablename__ = "idempotency_keys"
    
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True)
    entity_type: Mapped[str] = mapped_column(String(255))
    entity_id: Mapped[int]
    
    @staticmethod
    def generate() -> str:
        """Generate new idempotency key."""
        return str(uuid4())


class IdempotentService:
    """Service with idempotent operations."""
    
    def __init__(self, session):
        self.session = session
    
    def create_with_idempotency(
        self,
        entity_class,
        idempotency_key: str,
        **entity_data
    ):
        """Create entity only if idempotency key doesn't exist.
        
        Args:
            entity_class: ORM model class
            idempotency_key: Unique operation identifier
            **entity_data: Entity attributes
            
        Returns:
            Existing or newly created entity
        """
        # Check existing
        existing_key = self.session.query(IdempotencyKeyRecord).filter(
            IdempotencyKeyRecord.idempotency_key == idempotency_key
        ).first()
        
        if existing_key:
            # Return existing entity
            return self.session.query(entity_class).get(existing_key.entity_id)
        
        # Create new entity
        entity = entity_class(**entity_data)
        self.session.add(entity)
        self.session.flush()
        
        # Record idempotency key
        key_record = IdempotencyKeyRecord(
            idempotency_key=idempotency_key,
            entity_type=entity_class.__name__,
            entity_id=entity.id
        )
        self.session.add(key_record)
        self.session.commit()
        
        return entity
```

#### 2.4 Update `.github/workflows/ci.yml` for Database Testing

Add job:

```yaml
test-database:
  name: Database Integration Tests
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
      run: poetry install --with dev,database
    
    - name: Run database tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost/test_db
      run: poetry run pytest tests/integration/ -v
```

---

## Phase 3: Ledger Patterns (P2) - 3-4 Days

### Goal
Provide complete ledger/accounting system template

### Deliverables (when `include_database="sqlalchemy_ledger"`)

#### 3.1 Ledger Models

`src/{{cookiecutter.project_slug}}/ledger/models.py`

```python
"""Double-entry bookkeeping models."""

from enum import Enum as PyEnum
from decimal import Decimal
from sqlalchemy import String, Enum, ForeignKey, CheckConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..db.models import BaseModel, SoftDeleteMixin


class AccountType(str, PyEnum):
    """Standard chart of accounts types."""
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"


class Account(BaseModel, SoftDeleteMixin):
    """Chart of accounts with hierarchy support."""
    
    __tablename__ = "accounts"
    
    code: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    account_type: Mapped[AccountType] = mapped_column(Enum(AccountType))
    
    # Hierarchy
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id"))
    
    # Relationships
    parent: Mapped["Account | None"] = relationship(
        "Account", remote_side=[id], backref="children"
    )
    
    __table_args__ = (
        Index('idx_account_type', 'account_type'),
        Index('idx_account_parent', 'parent_id'),
    )


class TransactionStatus(str, PyEnum):
    """Transaction states."""
    DRAFT = "draft"
    POSTED = "posted"
    REVERSED = "reversed"


class Transaction(BaseModel):
    """Double-entry transaction."""
    
    __tablename__ = "transactions"
    
    transaction_date: Mapped[datetime]
    description: Mapped[str] = mapped_column(String(500))
    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus), default=TransactionStatus.DRAFT
    )
    posted_at: Mapped[datetime | None]
    
    # Relationships
    entries: Mapped[list["JournalEntry"]] = relationship(
        "JournalEntry", back_populates="transaction"
    )
    
    __table_args__ = (
        Index('idx_transaction_date', 'transaction_date'),
        Index('idx_transaction_status', 'status'),
    )


class JournalEntry(BaseModel):
    """Individual debit or credit entry."""
    
    __tablename__ = "journal_entries"
    
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transactions.id"))
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    
    debit_amount: Mapped[Decimal | None] = mapped_column(
        sqlalchemy.DECIMAL(15, 2), nullable=True
    )
    credit_amount: Mapped[Decimal | None] = mapped_column(
        sqlalchemy.DECIMAL(15, 2), nullable=True
    )
    
    entry_date: Mapped[datetime]
    description: Mapped[str] = mapped_column(String(500))
    
    # Relationships
    transaction: Mapped["Transaction"] = relationship(
        "Transaction", back_populates="entries"
    )
    account: Mapped["Account"] = relationship("Account")
    
    __table_args__ = (
        CheckConstraint(
            "(debit_amount IS NOT NULL AND credit_amount IS NULL) OR "
            "(debit_amount IS NULL AND credit_amount IS NOT NULL)"
        ),
        Index('idx_je_account_date', 'account_id', 'entry_date'),
        Index('idx_je_transaction', 'transaction_id'),
    )
```

#### 3.2 Ledger Validators

`src/{{cookiecutter.project_slug}}/ledger/validators.py`

```python
"""Double-entry bookkeeping validation."""

from decimal import Decimal
from typing import List, Tuple
from .models import JournalEntry, Transaction


class LedgerValidator:
    """Validator for ledger integrity."""
    
    @staticmethod
    def validate_entry(entry: JournalEntry) -> List[str]:
        """Validate single journal entry.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Ensure exactly one amount
        has_debit = entry.debit_amount is not None
        has_credit = entry.credit_amount is not None
        
        if not (has_debit ^ has_credit):
            errors.append("Entry must have debit XOR credit")
        
        # Check amounts are positive
        if has_debit and entry.debit_amount <= 0:
            errors.append("Debit amount must be positive")
        
        if has_credit and entry.credit_amount <= 0:
            errors.append("Credit amount must be positive")
        
        return errors
    
    @staticmethod
    def validate_transaction(transaction: Transaction) -> List[str]:
        """Validate complete transaction.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Need at least 2 entries
        if len(transaction.entries) < 2:
            errors.append("Transaction must have at least 2 entries")
            return errors
        
        # Calculate balance
        debits = sum((e.debit_amount or Decimal(0)) for e in transaction.entries)
        credits = sum((e.credit_amount or Decimal(0)) for e in transaction.entries)
        
        # Check balance (allow 0.01 for rounding)
        if abs(debits - credits) > Decimal('0.01'):
            errors.append(
                f"Transaction unbalanced: Debits={debits}, Credits={credits}"
            )
        
        return errors
```

#### 3.3 Ledger Services

`src/{{cookiecutter.project_slug}}/ledger/services.py`

```python
"""Ledger operations service."""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .models import Account, Transaction, JournalEntry, TransactionStatus
from .validators import LedgerValidator


class LedgerService:
    """Service for ledger operations."""
    
    def __init__(self, session: Session):
        self.session = session
        self.validator = LedgerValidator()
    
    def post_transaction(self, transaction: Transaction) -> None:
        """Post transaction to ledger.
        
        Args:
            transaction: Transaction to post
            
        Raises:
            ValueError: If validation fails
        """
        # Validate
        errors = self.validator.validate_transaction(transaction)
        if errors:
            raise ValueError(f"Transaction validation failed: {'; '.join(errors)}")
        
        # Post
        transaction.status = TransactionStatus.POSTED
        transaction.posted_at = datetime.utcnow()
        self.session.add(transaction)
    
    def reverse_transaction(self, transaction: Transaction) -> Transaction:
        """Reverse (void) a posted transaction.
        
        Args:
            transaction: Transaction to reverse
            
        Returns:
            New reversed transaction with opposite entries
            
        Raises:
            ValueError: If transaction is not posted
        """
        if transaction.status != TransactionStatus.POSTED:
            raise ValueError(f"Can only reverse posted transactions")
        
        # Mark original as reversed
        transaction.status = TransactionStatus.REVERSED
        
        # Create reversal with opposite entries
        reversal = Transaction(
            transaction_date=datetime.utcnow(),
            description=f"Reversal of transaction {transaction.id}",
            status=TransactionStatus.POSTED,
            posted_at=datetime.utcnow()
        )
        
        for entry in transaction.entries:
            reversal_entry = JournalEntry(
                account_id=entry.account_id,
                debit_amount=entry.credit_amount,
                credit_amount=entry.debit_amount,
                entry_date=datetime.utcnow(),
                description=f"Reversal of entry {entry.id}"
            )
            reversal.entries.append(reversal_entry)
        
        self.session.add(reversal)
        return reversal
    
    def get_account_balance(
        self,
        account: Account,
        as_of_date: datetime | None = None
    ) -> Decimal:
        """Get account balance as of date.
        
        Args:
            account: Account to calculate balance for
            as_of_date: Date for balance calculation (None = current)
            
        Returns:
            Account balance (positive for normal balance)
        """
        query = select(
            func.sum(JournalEntry.debit_amount).label('debits'),
            func.sum(JournalEntry.credit_amount).label('credits')
        ).where(
            JournalEntry.account_id == account.id,
            Transaction.status == TransactionStatus.POSTED
        ).select_from(JournalEntry).join(Transaction)
        
        if as_of_date:
            query = query.where(JournalEntry.entry_date <= as_of_date)
        
        result = self.session.execute(query).first()
        
        debits = result.debits or Decimal(0)
        credits = result.credits or Decimal(0)
        
        # Balance based on account type normal balance
        if account.account_type.value in ['asset', 'expense']:
            return debits - credits
        else:
            return credits - debits
```

#### 3.4 Ledger Reports

`src/{{cookiecutter.project_slug}}/ledger/reports.py`

```python
"""Financial report generation."""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .models import Account, JournalEntry, Transaction, TransactionStatus, AccountType
from .services import LedgerService


class FinancialReports:
    """Generate financial statements."""
    
    def __init__(self, session: Session):
        self.session = session
        self.ledger_service = LedgerService(session)
    
    def trial_balance(self, as_of_date: datetime | None = None) -> dict:
        """Generate trial balance.
        
        Returns:
            Dict of {account_code: balance}
        """
        accounts = self.session.query(Account).filter(
            Account.is_deleted == False
        ).all()
        
        return {
            account.code: self.ledger_service.get_account_balance(
                account, as_of_date
            )
            for account in accounts
        }
    
    def income_statement(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """Generate income statement (P&L).
        
        Returns:
            Dict with revenue, expenses, net_income
        """
        # Get income and expense accounts
        query = select(
            Account.account_type,
            func.sum(JournalEntry.debit_amount).label('debits'),
            func.sum(JournalEntry.credit_amount).label('credits')
        ).select_from(Account).join(
            JournalEntry, Account.id == JournalEntry.account_id
        ).join(Transaction, JournalEntry.transaction_id == Transaction.id).where(
            Account.account_type.in_(['income', 'expense']),
            JournalEntry.entry_date.between(start_date, end_date),
            Transaction.status == TransactionStatus.POSTED
        ).group_by(Account.account_type)
        
        results = self.session.execute(query).all()
        
        revenue = Decimal(0)
        expenses = Decimal(0)
        
        for result in results:
            if result.account_type == AccountType.INCOME:
                revenue += (result.credits or Decimal(0)) - (result.debits or Decimal(0))
            elif result.account_type == AccountType.EXPENSE:
                expenses += (result.debits or Decimal(0)) - (result.credits or Decimal(0))
        
        return {
            'revenue': revenue,
            'expenses': expenses,
            'net_income': revenue - expenses
        }
```

#### 3.5 Documentation

Create `docs/guides/ledger-patterns.md` with:
- Double-entry bookkeeping explanation
- Account hierarchy guide
- Transaction lifecycle
- Report generation examples

---

## Detailed File Checklist

### Phase 1 (Foundation)

- [ ] Update `cookiecutter.json` with `include_database` option
- [ ] Create `src/{{cookiecutter.project_slug}}/db/base.py`
- [ ] Create `src/{{cookiecutter.project_slug}}/db/models.py`
- [ ] Create `src/{{cookiecutter.project_slug}}/db/repository.py`
- [ ] Create `src/{{cookiecutter.project_slug}}/db/unit_of_work.py`
- [ ] Create `src/{{cookiecutter.project_slug}}/db/__init__.py`
- [ ] Update `core/config.py` with DatabaseSettings
- [ ] Update `pyproject.toml` dependencies
- [ ] Create database fixtures in `tests/conftest.py`
- [ ] Add database documentation to `docs/guides/`
- [ ] Create example models in `docs/examples/`

### Phase 2 (Advanced)

- [ ] Setup Alembic scaffolding
- [ ] Create `pydantic_models.py` with financial validators
- [ ] Create `db/idempotency.py` for idempotency keys
- [ ] Update CI workflow for database testing
- [ ] Add database tests to `tests/integration/`
- [ ] Create migration documentation
- [ ] Add performance tuning guide

### Phase 3 (Ledger)

- [ ] Create `ledger/models.py` with double-entry schema
- [ ] Create `ledger/validators.py` with balance validation
- [ ] Create `ledger/services.py` with ledger operations
- [ ] Create `ledger/reports.py` with financial statements
- [ ] Create `tests/fixtures/ledger_fixtures.py` with sample CoA
- [ ] Create ledger documentation
- [ ] Create ledger integration tests

---

## Testing Strategy

### Unit Tests (Phase 1)
```bash
tests/unit/db/
├── test_repository.py         # Repository CRUD operations
├── test_unit_of_work.py       # Transaction coordination
└── test_models.py             # Model validation
```

### Integration Tests (Phase 2)
```bash
tests/integration/
├── test_database_operations.py # Multi-repository flows
├── test_migrations.py          # Migration up/down
└── test_idempotency.py         # Duplicate prevention
```

### Ledger Tests (Phase 3)
```bash
tests/integration/ledger/
├── test_posting.py            # Transaction posting
├── test_balance.py            # Account balance calculations
├── test_reports.py            # Financial statements
└── test_reversals.py          # Transaction reversals
```

---

## Success Criteria

### Phase 1 Complete When:
- [ ] All 4 database options work without errors
- [ ] Fixtures pass and reduce test setup 20%
- [ ] 80%+ test coverage on db module
- [ ] Documentation covers Repository and UnitOfWork patterns

### Phase 2 Complete When:
- [ ] Alembic migrations generate and test successfully
- [ ] Pydantic validators prevent invalid financial data
- [ ] Idempotency keys prevent duplicate inserts
- [ ] CI/CD tests database operations successfully

### Phase 3 Complete When:
- [ ] Double-entry transactions balance correctly
- [ ] State machine enforces draft→posted→reversed flow
- [ ] Reports (trial balance, P&L) calculate correctly
- [ ] All ledger tests pass with 85%+ coverage

---

## Migration Path for Existing Users

1. **New projects**: Use `include_database` option
2. **Existing projects**: Can manually add `db/` module after generating
3. **Documentation**: Include migration guide for existing projects

---

## Estimated Timeline

- **Phase 1**: 15-20 hours (experienced Python dev)
- **Phase 2**: 12-16 hours
- **Phase 3**: 10-14 hours
- **Total**: ~40-50 hours of development

---

## Quality Gates

Before merging each phase:

1. **Code Quality**:
   - ✓ Ruff formatting passes
   - ✓ MyPy type checking (strict on src/)
   - ✓ Docstrings 90%+ coverage

2. **Testing**:
   - ✓ 80%+ line coverage
   - ✓ All database tests pass
   - ✓ CI/CD workflow succeeds

3. **Documentation**:
   - ✓ API docs complete
   - ✓ Architecture diagrams included
   - ✓ Usage examples provided

---

