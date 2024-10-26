
from pipecat.frames.frames import Frame, ImageRawFrame, TextFrame, VisionImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class VisionImageFrameAggregator(FrameProcessor):
    """This aggregator waits for a consecutive TextFrame and an
    ImageFrame. After the ImageFrame arrives it will output a VisionImageFrame.

    >>> from pipecat.pipeline.frames import ImageFrame

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         print(frame)

    >>> aggregator = VisionImageFrameAggregator()
    >>> asyncio.run(print_frames(aggregator, TextFrame("What do you see?")))
    >>> asyncio.run(print_frames(aggregator, ImageFrame(image=bytes([]), size=(0, 0))))
    VisionImageFrame, text: What do you see?, image size: 0x0, buffer size: 0 B

    """

    def __init__(self):
        super().__init__()
        self._describe_text = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self._describe_text = frame.text
        elif isinstance(frame, ImageRawFrame):
            if self._describe_text:
                frame = VisionImageRawFrame(
                    text=self._describe_text,
                    image=frame.image,
                    size=frame.size,
                    format=frame.format)
                await self.push_frame(frame)
                self._describe_text = None
        else:
            await self.push_frame(frame, direction)
