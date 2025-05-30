import hashlib

from loguru import logger


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
