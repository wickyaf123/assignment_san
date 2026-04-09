"""
Neo4j driver factory with env-based configuration for ViddhiAI.

Provides a module-level singleton driver to avoid reconnecting on every call.
Environment variables (loaded from .env via python-dotenv):
    NEO4J_URI      — bolt/neo4j+s URI for the database
    NEO4J_USER     — database username (usually "neo4j")
    NEO4J_PASSWORD — database password
"""

from __future__ import annotations

import logging
import os

import neo4j
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

_driver: neo4j.Driver | None = None


def get_driver() -> neo4j.Driver:
    """Return the singleton Neo4j driver, creating it on first call.

    Reads NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD from environment.
    Raises ValueError if any env var is missing.
    """
    global _driver

    if _driver is not None:
        return _driver

    uri = os.environ.get("NEO4J_URI")
    if not uri:
        raise ValueError("Missing required env var: NEO4J_URI")

    user = os.environ.get("NEO4J_USER")
    if not user:
        raise ValueError("Missing required env var: NEO4J_USER")

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise ValueError("Missing required env var: NEO4J_PASSWORD")

    logger.info("Connecting to Neo4j at %s as %s", uri, user)
    _driver = neo4j.GraphDatabase.driver(
        uri,
        auth=(user, password),
        max_connection_pool_size=20,
        connection_acquisition_timeout=10,
        max_connection_lifetime=300,
    )
    return _driver


def close_driver() -> None:
    """Close the singleton driver and reset it to None."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")
