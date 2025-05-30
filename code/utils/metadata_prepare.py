from enum import Enum


class AcceptedDTypes(Enum):
    STR = str
    INT = int
    FLOAT = float
    BOOL = bool


def check_dtype(value) -> bool:
    """
    Check if the value is of an accepted data type.
    :param value: The value to check
    :return: Whether the value is of an accepted data type
    """
    for dtype in AcceptedDTypes:
        if isinstance(value, dtype.value):
            return True
    return False


def cast_metadata(
    metadata_dict: dict, conv_id: str
) -> dict[str, str | int | float | bool]:
    """
    Cast the metadata dictionary to the correct types.
    :param metadata_dict: The metadata dictionary
    :param conv_id: The conversation ID
    :return: The casted metadata dictionary
    """
    res = dict()
    for key, value in metadata_dict.items():
        if check_dtype(value):
            res[key] = value
        else:
            res[key] = str(value)
    res["conv_id"] = conv_id
    return res
