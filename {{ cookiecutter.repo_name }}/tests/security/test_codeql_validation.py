#!/usr/bin/env python3
"""
Test file for CodeQL Python security scanning validation.
This file contains intentional security vulnerabilities to test CodeQL detection.
These patterns should be detected by CodeQL's security-extended queries.

WARNING: This file contains intentional security vulnerabilities for testing purposes only.
Do not use these patterns in production code.
"""

import hashlib
import pickle
import sqlite3
import subprocess


def hardcoded_secret_test():
    """Test hardcoded secrets detection."""
    # CodeQL should detect this hardcoded secret
    api_key = "sk-1234567890abcdef1234567890abcdef"  # nosec - intentional for testing
    password = "admin123"  # nosec - intentional for testing
    token = "ghp_1234567890abcdef1234567890abcdef123456"  # nosec - intentional for testing
    return api_key, password, token


def sql_injection_test(user_input):
    """Test SQL injection vulnerability detection."""
    # CodeQL should detect this SQL injection vulnerability
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Vulnerable SQL query construction
    query = f"SELECT * FROM users WHERE name = '{user_input}'"  # nosec - intentional for testing
    cursor.execute(query)

    return cursor.fetchall()


def command_injection_test(user_input):
    """Test command injection vulnerability detection."""
    # CodeQL should detect this command injection vulnerability
    command = f"ls {user_input}"  # nosec - intentional for testing
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )  # nosec - intentional for testing
    return result.stdout


def path_traversal_test(filename):
    """Test path traversal vulnerability detection."""
    # CodeQL should detect this path traversal vulnerability
    file_path = f"/var/www/uploads/{filename}"  # nosec - intentional for testing

    # Vulnerable file access without validation
    with open(file_path) as f:  # nosec - intentional for testing
        return f.read()


def insecure_deserialization_test(data):
    """Test insecure deserialization vulnerability detection."""
    # CodeQL should detect this insecure deserialization
    return pickle.loads(data)  # nosec - intentional for testing


def weak_cryptography_test(data):
    """Test weak cryptography detection."""
    # CodeQL should detect weak hash algorithm usage
    return hashlib.md5(data.encode()).hexdigest()  # nosec - intentional for testing


def unsafe_eval_test(user_code):
    """Test unsafe eval usage detection."""
    # CodeQL should detect unsafe eval usage
    return eval(user_code)  # nosec - intentional for testing


if __name__ == "__main__":
    # These function calls are for testing purposes only
    # They should trigger CodeQL security alerts

    print("Testing hardcoded secrets...")
    secrets = hardcoded_secret_test()

    print("Testing SQL injection...")
    sql_result = sql_injection_test("admin'; DROP TABLE users; --")

    print("Testing command injection...")
    cmd_result = command_injection_test("; rm -rf /")

    print("Testing path traversal...")
    try:
        file_content = path_traversal_test("../../../etc/passwd")
    except FileNotFoundError:
        print("File not found (expected)")

    print("Testing insecure deserialization...")
    try:
        unsafe_data = b"test data"
        result = insecure_deserialization_test(unsafe_data)
    except Exception as e:
        print(f"Deserialization error (expected): {e}")

    print("Testing weak cryptography...")
    weak_hash = weak_cryptography_test("sensitive data")

    print("Testing unsafe eval...")
    try:
        eval_result = unsafe_eval_test("print('Hello World')")
    except Exception as e:
        print(f"Eval error: {e}")

    print("CodeQL validation test completed.")
