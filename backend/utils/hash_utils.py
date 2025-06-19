import hashlib

from loguru import logger

# TODO : hashing contents to generate database IDs is a legacy
# of very very early versions of the project.
# Hashing message and metadata together lets us avoid collisions,
# but breaks down when we slightly change the metadata formatting and extraction.
# This should be refactored to use a more robust ID generation strategy in the future.
# Plan is to eventually switch to UUIDs or a similar approach that guarantees uniqueness without
# relying on the content of the message.

# Below, an AI-generated explanation of the hybrid message identification system

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID MESSAGE IDENTIFICATION SYSTEM: HASH + UUID
#
# Motivation:
# In this system, every message in the Chroma database is identified by a *deterministic*
# content-based hash (generated from the message text + metadata), which serves as the
# primary key for database upserts. This allows us to avoid duplicate entries when
# re-running ingestion scripts or syncing with external sources — as long as the
# message and its relevant metadata stay consistent, the hash will too.
#
# However, hashes are fragile: even small changes in formatting, metadata fields,
# or whitespace can result in a completely different hash. Over time, this makes
# maintenance and external referencing brittle, especially if you change the
# metadata schema (which is *inevitable* in a long-running project).
#
# To solve this, we assign a *secondary*, persistent, and non-deterministic UUID
# to each message at the time of first ingestion. This UUID is stored alongside
# the message in the database as part of its metadata. This creates a stable
# identity reference that can be reused for things like:
#  - dynamic tagging systems
#  - semantic search results
#  - external references in metadata files
#  - long-term exports or cross-database linking
#
# The system works like this:
#
# 1. On message ingestion:
#    - A deterministic hash is computed from the message content + metadata.
#    - This hash becomes the Chroma message ID (used for upserting).
#    - If this hash is not yet known, a new UUID is generated and associated with it.
#    - The UUID is stored *in the message's metadata*, for long-term use.
#
# 2. When re-ingesting:
#    - The hash is re-computed from the message.
#    - If the hash already exists, we *reuse* the UUID from the previous ingestion.
#    - If not, we treat the message as new and assign a new UUID.
#
# This system gives you:
#    ✅ Duplicate prevention via deterministic IDs
#    ✅ Long-term message identity via UUID
#    ✅ Flexibility to update schemas without breaking identity
#
# This approach future-proofs your database and allows for external tools
# (like classification scripts, frontend queries, or semantic exports)
# to operate robustly, even if message metadata or structure evolves.
#
# Pro tip:
# You should persist the hash → UUID mapping (e.g., as a JSON file)
# to avoid regenerating UUIDs across runs. This lets you restore identity
# after crashes or refactors.
#
# Example:
# {
#     "fa4c893b41e66f...": "8b4fbd4e-054b-11ef-9d4a-4f92f7f460dd",
#     ...
# }
#
# This whole thing may sound overkill, but for large-scale chat archives
# and metadata workflows, it’s a lifesaver.
# ─────────────────────────────────────────────────────────────────────────────


def hash_message(message: str) -> str:
    """
    Hash an exchange.
    :param exchange: The exchange json string
    :return: The hashed exchange
    """
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def hash_set_length(text: bytes, length: int = 8) -> str:
    """
    Hash a string and return the first `length` characters.
    :param text: The text to hash -- encoded in UTF-8
    :param length: The desired length of the hash
    :return: The hashed string
    """
    if length < 1:
        raise ValueError("The desired length should be greater than 0")
    elif length > 64:
        raise ValueError("The desired length should be less than or equal to 64")

    return hashlib.sha256(text).hexdigest()[:length]


def pad_int(number: int, length: int = 3) -> str:
    """
    Pad an integer with zeros to a certain length.
    :param number: The number to pad
    :param length: The desired length
    :return: The padded number
    """
    if length < 1:
        raise ValueError("The desired length should be greater than 0")
    elif len(str(number)) > length:
        logger.warning(f"The number {number} is longer than the desired length ({length})")
        length = len(str(number))
    return str(number).zfill(length)
