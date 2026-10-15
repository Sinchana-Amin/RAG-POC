"""Configure Python HTTP clients to use the machine or bundled CA certificates."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_ssl_trust() -> None:
    """Configure requests to trust the bundled CA or the platform certificate store."""
    # Honor the bundled CA file when present; otherwise retain the platform defaults.
    bundle_path = Path(__file__).resolve().parent / "certs" / "pip-ca-bundle.pem"
    if bundle_path.exists():
        bundle = str(bundle_path)
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", bundle)

    try:
        import truststore

        truststore.inject_into_ssl()
        logger.debug("Injected platform certificate trust into Python SSL.")
    except ImportError:
        logger.debug("truststore is not installed; using the configured/default CA certificates.")
    except Exception:
        logger.warning(
            "Could not inject platform certificate trust; continuing with configured/default CA certificates.",
            exc_info=True,
        )


configure_ssl_trust()
