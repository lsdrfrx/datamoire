from abc import ABC, abstractmethod
from typing import Union

class Storage(ABC):
    @abstractmethod
    def create_board(name: str) -> bool: pass

    @abstractmethod
    def write(board_name: str, index: str, value: Union[int, float]) -> None: pass