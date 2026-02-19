"""Global test configuration."""

import os

# Disable SSL in tests so TestClient works over plain HTTP
os.environ["STT_SSL_ENABLED"] = "false"

# Use a writable local path for test runs outside Docker.
os.environ["OS_VOICE_LIBRARY_PATH"] = "./data/voices"
