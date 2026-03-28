import math
from collections.abc import Sequence


def normalize_l2(vector: Sequence[float]) -> list[float]:
    values = [float(component) for component in vector]
    norm = math.sqrt(sum(component * component for component in values))
    if norm == 0.0:
        return values
    return [component / norm for component in values]
