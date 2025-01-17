import sys
import os

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.method.basemethod import BaseMethod
from src.method.zero_shot import ZeroShot

methods = {
    "zero_shot": ZeroShot,
}

method_dict = {}
method_dict.update(methods)

def get_method(method_name, *args, **kwargs) -> BaseMethod:
    return method_dict[method_name](method_name=method_name, *args, **kwargs)