from typing import Union

from .storages._base import Storage

class TorchBoard:
    def __init__(self, storage: Storage):
        self.storage = storage

    def write(self, board_name: str, index: str, value: Union[int, float]) -> None:
        self.storage.write(board_name, index, value)

    def create_board(self, board_name: str) -> bool:
        return self.storage.create_board(board_name)