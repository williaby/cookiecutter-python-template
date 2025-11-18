# Notification Service

Central notification service for email, SMS, and push notifications.

## 🎯 Purpose

Unified notification delivery across all projects with templates, tracking, and compliance.

## ✨ Features

- **Multi-Channel**: Email (SMTP/SendGrid), SMS (Twilio), Push (FCM/APNS)
- **Template Management**: Jinja2 templates with versioning
- **Delivery Tracking**: Track sent, delivered, opened, clicked
- **Rate Limiting**: Prevent spam, respect quotas
- **Queue Management**: Background job processing with retry
- **Unsubscribe Handling**: GDPR-compliant opt-out
- **A/B Testing**: Test different message variants
- **Scheduling**: Send at specific times/timezones

## 📦 Code to Migrate from Template

### From `{{cookiecutter.project_slug}}/src/*/utils/`

**Email sending code** - Look for:
```python
# MIGRATE THIS TO CENTRAL SERVICE
import smtplib
from email.mime.text import MIMEText

def send_email(to: str, subject: str, body: str):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['To'] = to
    # ... SMTP configuration
```

**Replace with**:
```python
from notification_client import NotificationClient

notif = NotificationClient()
await notif.send_email(
    to="user@example.com",
    template="welcome_email",
    context={"username": "John"}
)
```

### Benefits of Centralization

- **Templates**: Consistent branding across all projects
- **Deliverability**: Centralized reputation management
- **Compliance**: Centralized unsubscribe handling
- **Tracking**: Unified delivery analytics
- **Cost**: Negotiate better rates with providers

## 🔌 Client SDK

```python
from notification_client import NotificationClient

client = NotificationClient(url="https://notifications.your-domain.com")

# Email
await client.send_email(
    to="user@example.com",
    template="welcome",
    context={"name": "John"},
    tags=["onboarding"]
)

# SMS
await client.send_sms(
    to="+1234567890",
    template="verification_code",
    context={"code": "123456"}
)

# Push notification
await client.send_push(
    user_id="user-123",
    title="New message",
    body="You have a new message",
    data={"message_id": "msg-456"}
)

# Batch send
await client.send_batch(
    template="monthly_newsletter",
    recipients=[{"email": "user1@example.com", "name": "Alice"}, ...],
    tags=["newsletter", "marketing"]
)
```

## 📋 Migration Candidates

Projects with email/SMS code should migrate to this service.

---

**Status**: 🟡 Recommended
**Priority**: High (if sending notifications)
