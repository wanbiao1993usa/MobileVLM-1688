
from .bm1688_interface import Bm1688Model


def Model(file_path):
    return Bm1688Model(file_path)