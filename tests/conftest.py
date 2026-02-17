"""Global test configuration."""

import os

# Disable SSL in tests so TestClient works over plain HTTP
os.environ["STT_SSL_ENABLED"] = "false"
