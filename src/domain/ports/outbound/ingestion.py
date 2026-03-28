from typing import Protocol


class SocialSourceIngestionPort(Protocol):
    def collect(self, topic: str, max_items: int) -> list[dict[str, str]]: ...

    def normalize(self, raw_items: list[dict[str, str]]) -> list[dict[str, str]]: ...
