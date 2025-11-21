"""Financial utilities module."""

{% if cookiecutter.use_decimal_precision == "yes" -%}
CRITICAL: Always use Decimal for financial calculations to avoid floating-point errors.

Example:
    >>> from decimal import Decimal
    >>> price = Decimal('19.99')
    >>> quantity = Decimal('3')
    >>> total = price * quantity
    >>> # total is Decimal('59.97'), not 59.97000000000001

Never use float for money:
    >>> # WRONG - floating point errors
    >>> price = 19.99
    >>> quantity = 3
    >>> total = price * quantity  # 59.97000000000001

    >>> # CORRECT - using Decimal
    >>> from decimal import Decimal
    >>> price = Decimal('19.99')
    >>> total = price * Decimal('3')  # Decimal('59.97')
"""
{% endif -%}

{% if cookiecutter.use_decimal_precision == "yes" -%}
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Union

# Set default precision for financial calculations
getcontext().prec = 28  # Standard precision for financial calculations


def to_decimal(value: Union[str, int, float, Decimal]) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Value to convert (str, int, float, or Decimal)

    Returns:
        Decimal value with proper precision

    Raises:
        ValueError: If value cannot be converted to Decimal

    Examples:
        >>> to_decimal("19.99")
        Decimal('19.99')
        >>> to_decimal(100)
        Decimal('100')

    Warning:
        Passing float values may cause precision issues.
        Always prefer string or int inputs.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        # Convert to string first to avoid float precision issues
        return Decimal(str(value))
    return Decimal(value)


def round_currency(amount: Decimal, places: int = 2) -> Decimal:
    """Round currency amount to specified decimal places.

    Args:
        amount: Amount to round
        places: Number of decimal places (default: 2)

    Returns:
        Rounded Decimal amount

    Examples:
        >>> round_currency(Decimal('19.995'))
        Decimal('20.00')
        >>> round_currency(Decimal('19.994'))
        Decimal('19.99')
    """
    quantizer = Decimal('0.1') ** places
    return amount.quantize(quantizer, rounding=ROUND_HALF_UP)


def validate_positive(amount: Decimal, field_name: str = "amount") -> None:
    """Validate that amount is positive.

    Args:
        amount: Amount to validate
        field_name: Name of field for error messages

    Raises:
        ValueError: If amount is not positive

    Examples:
        >>> validate_positive(Decimal('10.00'))  # OK
        >>> validate_positive(Decimal('-5.00'))  # Raises ValueError
        Traceback (most recent call last):
            ...
        ValueError: amount must be positive, got -5.00
    """
    if amount <= 0:
        raise ValueError(f"{field_name} must be positive, got {amount}")


def calculate_percentage(
    amount: Decimal,
    percentage: Decimal,
    round_places: int = 2
) -> Decimal:
    """Calculate percentage of amount.

    Args:
        amount: Base amount
        percentage: Percentage as Decimal (e.g., Decimal('10') for 10%)
        round_places: Decimal places for result

    Returns:
        Calculated percentage amount

    Examples:
        >>> calculate_percentage(Decimal('100'), Decimal('10'))
        Decimal('10.00')
        >>> calculate_percentage(Decimal('99.99'), Decimal('7.5'))
        Decimal('7.50')
    """
    result = (amount * percentage) / Decimal('100')
    return round_currency(result, round_places)


{% if cookiecutter.include_audit_logging == "yes" -%}
def format_audit_amount(amount: Decimal) -> str:
    """Format amount for audit logging.

    Args:
        amount: Amount to format

    Returns:
        Formatted string with currency symbol

    Examples:
        >>> format_audit_amount(Decimal('1234.56'))
        '$1,234.56'
    """
    # Format with thousand separators and 2 decimal places
    return f"${amount:,.2f}"
{% endif -%}
{% endif -%}
