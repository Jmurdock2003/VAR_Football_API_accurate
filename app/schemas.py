from pydantic import BaseModel
from typing import List, Dict, Any

class Track(BaseModel):
    id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    cls: int

class Payload(BaseModel):
    frame_id: int
    tracks: List[Track]
    teams: Dict[int, int]
    cam_vec: Any