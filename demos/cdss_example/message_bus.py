"""Standalone message bus for streaming tokens.

This module provides a shared message queue with zero dependencies to avoid circular imports.
"""
import asyncio

# Global message bus queue for streaming tokens (thread-safe)
# Use a large maxsize to prevent dropping tokens
message_queue = asyncio.Queue(maxsize=10000)

