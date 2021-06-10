from pydantic import BaseModel
from typing import Optional


class RangeClusterNumbers(BaseModel):
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None


class ClusterNumber(BaseModel):
    k: int
