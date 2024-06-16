from enum import Enum, auto

# There is used only FULL mode, other modes are ready for app modifications


class IcrType(Enum):
    FULL = auto()
    SUN_ONLY = auto()
    VEGETATION_ONLY = auto()
