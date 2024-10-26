
import aiohttp
import time
import re
import json

from typing import AsyncGenerator

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AsyncAIService, TTSService

import re
from loguru import logger

# See .env.example for Deepgram configuration needed
try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        LiveOptions,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`. Also, set `DEEPGRAM_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class DeepgramTTSService(TTSService):

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        api_key: str,
        voice: str = "aura-helios-en",
        base_url: str = "https://api.deepgram.com/v1/speak",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._voice = voice
        self._api_key = api_key
        self._aiohttp_session = aiohttp_session
        self._base_url = base_url

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        base_url = self._base_url
        request_url = f"{base_url}?model={self._voice}&encoding=linear16&container=none&sample_rate=16000"
        headers = {"authorization": f"token {self._api_key}"}
        body = {"text": text}

        try:
            await self.start_ttfb_metrics()
            async with self._aiohttp_session.post(
                request_url, headers=headers, json=body
            ) as r:
                if r.status != 200:
                    response_text = await r.text()
                    # If we get a a "Bad Request: Input is unutterable", just print out a debug log.
                    # All other unsuccesful requests should emit an error frame. If not specifically
                    # handled by the running PipelineTask, the ErrorFrame will cancel the task.
                    if "unutterable" in response_text:
                        logger.debug(f"Unutterable text: [{text}]")
                        return

                    logger.error(
                        f"{self} error getting audio (status: {r.status}, error: {response_text})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status}, error: {response_text})"
                    )
                    return

                async for data in r.content:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
                    yield frame
        except Exception as e:
            logger.exception(f"{self} exception: {e}")


class DeepgramSTTService(AsyncAIService):
    def __init__(
        self,
        *,
        api_key: str,
        url: str = "",
        live_options: LiveOptions = LiveOptions(
            encoding="linear16",
            language="en-US",
            #  language="hi",
            model="nova-2-conversationalai",
            #  model="nova-2-general",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            smart_format=True,
        ),
        language: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._live_options = live_options

        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(url=url, options={"keepalive": "true"}),
        )
        self._connection = self._client.listen.asynclive.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
        self._language = language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            await self._connection.send(frame.audio)
        else:
            await self.queue_frame(frame, direction)

    async def start(self, frame: StartFrame):
        if await self._connection.start(self._live_options):
            logger.debug(f"{self}: Connected to Deepgram")
        else:
            logger.error(f"{self}: Unable to connect to Deepgram")

    async def stop(self, frame: EndFrame):
        await self._connection.finish()

    async def cancel(self, frame: CancelFrame):
        await self._connection.finish()

    async def _on_message(self, *args, **kwargs):
        result = kwargs["result"]
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        if len(transcript) > 0:

            # log the transcript
            logger.debug(f"Transcript: {transcript}")

            if is_final:
                # updated_transcript = await self._process_punctuation(transcript)
                _transcript = await self._process_numbers(transcript)
                
                # remove spaces and commas between numbers
                _updated_number = self._remove_spaces_followed_by_number(_transcript)
                                
                _entity_transcript = await self.process_entities(_updated_number, self._language)

                # log the updated transcript
                logger.debug(f"Updated transcript: {_entity_transcript}")

                await self.queue_frame(
                    TranscriptionFrame(
                        _entity_transcript, "", int(time.time_ns() / 1000000)
                    )
                )
            else:
                await self.queue_frame(
                    InterimTranscriptionFrame(
                        transcript, "", int(time.time_ns() / 1000000)
                    )
                )

    async def _process_punctuation(self, transcript: str):
        # if there are . at the end of the transcript, remove them
        if transcript and transcript[-1] == ".":
            transcript = transcript[:-1]

        return transcript

    async def process_entities(self, transcript: str, language: str):
        try:

            # log the transcript before processing
            logger.debug(f"Transcript before processing: {transcript}")

            data = json.dumps(
                {
                    "query": transcript,
                    "language": language,
                    "domain": "generic",
                    "entities": ["number", "exception"],
                }
            )

            headers = {"Content-Type": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://20.193.190.166:80/get_entities",
                    data=data,
                    headers=headers,
                ) as response:
                    response_json = await response.json()
                    # translatedValue = response_json["responseList"][0]["outString"][0]

                    display_text = response_json["display_text"]

                    # log the translated value
                    logger.debug(f"Updated text value: {display_text}")

                    return display_text
        except Exception as e:
            logger.debug(f"Exception in ITN api {e}")
            return transcript

    async def _process_numbers(self, transcript: str):
        # शून्य, एक, दो, तीन, चार, पांच, छह, सात, आठ, नौ
        # शून्य, तीन, चार, पांच, छह, सात, आठ, नौ - to check theses, other words have different meanings too
        if (
            "शून्य" in transcript
            or "तीन" in transcript
            or "चार" in transcript
            or "पांच" in transcript
            or "छह" in transcript
            or "सात" in transcript
            or "आठ" in transcript
            or "नौ" in transcript
        ):
            # replace word to numbers
            transcript = transcript.replace("शून्य", "0")
            transcript = transcript.replace("एक", "1")
            transcript = transcript.replace("दो", "2")
            transcript = transcript.replace("तीन", "3")
            transcript = transcript.replace("चार", "4")
            transcript = transcript.replace("पांच", "5")
            transcript = transcript.replace("छह", "6")
            transcript = transcript.replace("सात", "7")
            transcript = transcript.replace("आठ", "8")
            transcript = transcript.replace("नौ", "9")

            # also replace word+space to get continuos numbers as one
            transcript = transcript.replace("शून्य ", "0")
            transcript = transcript.replace("एक ", "1")
            transcript = transcript.replace("दो ", "2")
            transcript = transcript.replace("तीन ", "3")
            transcript = transcript.replace("चार ", "4")
            transcript = transcript.replace("पांच ", "5")
            transcript = transcript.replace("छह ", "6")
            transcript = transcript.replace("सात ", "7")
            transcript = transcript.replace("आठ ", "8")
            transcript = transcript.replace("नौ ", "9")

            transcript = self._remove_comma_followed_by_number(transcript)
            transcript = self._remove_spaces_followed_by_number(transcript)

        return transcript

    def _remove_spaces_followed_by_number(self, transcript):
        # function uses a lookbehind assertion (?<=\d) to ensure the space is preceded by a digit
        # a lookahead assertion (?=\d) to ensure the space is followed by a digit.
        # also removes spaces between multiple numbers joined together
        return re.sub(r"(?<=\d)\s(?=\d)", "", transcript)

    def _remove_comma_followed_by_number(self, transcript):
        # function uses a lookbehind assertion (?<=\d) to ensure the comma is preceded by a digit
        # a lookahead assertion (?=\d) to ensure the comman is followed by a digit or multiple spaces.
        return re.sub(r"(?<=\d),\s*(?=\d)", "", transcript)
