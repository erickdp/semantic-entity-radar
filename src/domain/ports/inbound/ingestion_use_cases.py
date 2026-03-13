from typing import Protocol


class StartIngestionUseCasePort(Protocol):
    def execute(self, topic: str, max_items: int) -> tuple[str, str]: ...
