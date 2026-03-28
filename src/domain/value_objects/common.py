from dataclasses import dataclass


@dataclass(frozen=True)
class TopicText:
    value: str

    def __post_init__(self) -> None:
        cleaned = self.value.strip()
        if len(cleaned) < 3:
            msg = "query topic must be at least 3 characters"
            raise ValueError(msg)


@dataclass(frozen=True)
class SourceLimit:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1 or self.value > 20:
            msg = "max sources must be between 1 and 20"
            raise ValueError(msg)
