
from abc import ABC, abstractmethod

from pipecat.frames.frames import Frame


class FrameSerializer(ABC):

    @abstractmethod
    def serialize(self, frame: Frame) -> str | bytes | None:
        pass

    @abstractmethod
    def deserialize(self, data: str | bytes) -> Frame | None:
        pass
    class Config:
        arbitrary_types_allowed = True
