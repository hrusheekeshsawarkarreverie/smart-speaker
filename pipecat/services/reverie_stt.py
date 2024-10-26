
import base64
import json
import time

from typing import Optional
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AsyncAIService
# from pipecat.utils.time import time_now_iso8601

from loguru import logger

# See .env.example for Gladia configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Gladia, you need to `pip install pipecat-ai[gladia]`. Also, set `GLADIA_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class ReverieSTTService(AsyncAIService):
    class InputParams(BaseModel):
        sample_rate: Optional[int] = 16000
        language: Optional[str] = "english"
        transcription_hint: Optional[str] = None
        endpointing: Optional[int] = 200
        prosody: Optional[bool] = None

    def __init__(self,
                 *,
                 api_key: str,
                 src_lang: str = "en",
                 domain: Optional[str] = "generic",
                 **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._src_lang = src_lang
        self._domain = domain
        # self._url = f"wss://revapi.reverieinc.com/stream?apikey={self._api_key}&appid=rev.stt_tts&appname=stt_stream&continuous=1&src_lang={self._src_lang}&punctuate=false&domain={self._domain}&silence=2&format=8k_int16"
        self._url = f"ws://20.193.177.219/stream?apikey={self._api_key}&appid=rev.stt_tts&appname=stt_stream&continuous=1&src_lang={self._src_lang}&punctuate=false&domain={self._domain}&silence=2&format=16k_int16&partials=false&timestamps=false"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            if self.is_websocket_open():
                await self._send_audio(frame)
            else:
                logger.warning("Websocket connection is not open. Skipping audio transmission.")
        else:
            await self.queue_frame(frame, direction)

    # function to check if the websocket connection is live
    def is_websocket_open(self):
        return self._websocket.open

    async def start(self, frame: StartFrame):
        # log that we are starting the websocket connection
        logger.info("Starting websocket connection")
        # log the connection url
        logger.info(f"Connection URL: {self._url}")
        
        self._websocket = await websockets.connect(self._url)
        if self._websocket.open:
            # log the connection is open
            logger.info("Websocket connection is open with url: " + self._url)
            
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
        # await self._setup_reverie()

    async def stop(self, frame: EndFrame):
        await self._websocket.close()

    async def cancel(self, frame: CancelFrame):
        await self._websocket.close()

    async def _send_audio(self, frame: AudioRawFrame):        
        await self._websocket.send(frame.audio)

    async def _receive_task_handler(self):
        async for message in self._websocket:
            utterance = json.loads(message)
            
            # # log the utterance
            # logger.info("--------------------")
            # logger.info("--------------------")
            
            final = utterance.get("final")
            display_text = utterance.get("display_text")
            cause = utterance.get("cause")
            
            if not display_text:
                continue
            
            
            if len(display_text) > 0:
                if cause == "silence detected":
                    # log that the utterance is final
                    logger.info(utterance)
                    logger.info(f"Final Utterance: {display_text}")
                    await self.queue_frame(TranscriptionFrame(display_text, "", int(time.time_ns() / 1000000)))
                else:
                    logger.info(f"Interim Utterance: {display_text}")
                    # await self.queue_frame(InterimTranscriptionFrame(display_text, "", int(time.time_ns() / 1000000)))
