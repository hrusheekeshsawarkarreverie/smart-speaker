from loguru import logger
import asyncio
import math
import struct
import time
from dataclasses import dataclass, field
from typing import List
import os
from abc import abstractmethod
from typing import AsyncGenerator, Awaitable, Callable
import asyncio

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    ErrorFrame,
    EndStreamFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMMessagesFrame,
)
import aiohttp
from pipecat.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.services.deepgram import DeepgramTTSService

from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
    OpenAILLMService,
    KrutrimLLMService,
    AzureOpenAILLMService,
)
from pipecat.services.ai_services import AIService
from twilio.rest import Client
import re
import requests
from fastapi import WebSocket
import json


class TTSService(AIService):
    def __init__(self, *, aggregate_sentences: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._aggregate_sentences: bool = aggregate_sentences
        self._current_sentence: str = ""

    # Converts the text to audio.
    @abstractmethod
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        pass

    # write method to get current sentence
    async def get_current_sentence(self):
        return self._current_sentence

    # write method to clear current sentence
    async def clear_current_sentence(self):
        self._current_sentence = ""

    async def say(self, text: str):
        await self.process_frame(TextFrame(text=text), FrameDirection.DOWNSTREAM)

    async def _process_text_frame(self, frame: TextFrame):
        text: str | None = None
        if not self._aggregate_sentences:
            text = frame.text
        else:
            self._current_sentence += frame.text
            if self._current_sentence.strip().endswith(
                (".", "?", "!")
            ) and not self._current_sentence.strip().endswith(
                ("Mr,", "Mrs.", "Ms.", "Dr.")
            ):
                text = self._current_sentence
                self._current_sentence = ""

        if text:
            # log the text before pushing the audio frames
            logger.debug(f"Text before generating audio: {text}")
            await self._push_tts_frames(text)

    async def _push_tts_frames(self, text: str):
        text = text.strip()
        if not text:
            return

        await self.push_frame(TTSStartedFrame())
        await self.start_processing_metrics()
        await self.process_generator(self.run_tts(text))
        await self.stop_processing_metrics()
        await self.push_frame(TTSStoppedFrame())
        # We send the original text after the audio. This way, if we are
        # interrupted, the text is not added to the assistant context.
        await self.push_frame(TextFrame(text))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self._process_text_frame(frame)
        elif isinstance(frame, StartInterruptionFrame):
            self._current_sentence = ""
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame) or isinstance(frame, EndFrame):
            self._current_sentence = ""
            await self._push_tts_frames(self._current_sentence)
            await self.push_frame(frame)
        else:
            await self.push_frame(frame, direction)


class ClearableDeepgramTTSService(DeepgramTTSService):
    def __init___(self, **kwargs):
        super().__init(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            self._current_sentence = ""


class ReverieOpenAILLMService(OpenAILLMService):
    def __init___(self, **kwargs):
        super().__init(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        

class ReverieAzureOpenAILLMService(AzureOpenAILLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)


class KrutrimOpenAILLMService(KrutrimLLMService):
    def __init___(self, **kwargs):
        super().__init(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)


# create ClearableElevenLabsTTSService class
class ClearableElevenLabsTTSService(ElevenLabsTTSService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # get the serializer and stream_id from the kwargs
        serializer = kwargs.get("serializer")
        stream_id = kwargs.get("stream_id")

        # print the stream_id and serializer
        logger.debug(f"stream_id: {stream_id}")
        logger.debug(f"serializer: {serializer}")

        self.serializer = serializer
        self.stream_id = stream_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # log that the process_frame method is called
        # logger.debug("process_frame method called for ClearableElevenLabsTTSService")

        # if isinstance(frame, StartInterruptionFrame):
        #     await self.clear_current_sentence()


@dataclass
class BufferedSentence:
    audio_frames: List[AudioRawFrame] = field(default_factory=list)
    text_frame: TextFrame = None


class TranscriptionTimingLogger(FrameProcessor):
    def __init__(self, avt):
        super().__init__()
        self.name = "Transcription"
        self._avt = avt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                elapsed = time.time() - self._avt.last_transition_ts
                logger.debug(f"Transcription TTF: {elapsed}")
                await self.push_frame(MetricsFrame(ttfb={self.name: elapsed}))

            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"Exception {e}")


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                logger.debug(f"Audio Transcription: [{frame.text}]")

            await self.push_frame(frame, direction)

        except Exception as e:
            logger.debug(f"Exception {e}")


class InterruptionHandler(FrameProcessor):
    def __init__(
        self,
        websocket_client: WebSocket,
        stream_sid: str,
        call_sid: str,
        call_provider: str,
        **kwargs,
    ):
        super().__init__()
        self.websocket_client = websocket_client
        self.stream_sid = stream_sid
        self.call_sid = call_sid
        self.last_text_frame_time = time.time()
        self.time_since_last_text_frame = 0
        self._call_provider = call_provider

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartInterruptionFrame):            
            if self._call_provider == "twilio":
                await self._clear_twilio_call_audio()
            elif self._call_provider == "exotel":
                await self._clear_exotel_call_audio()
            else:
                # Default to Twilio
                await self._clear_twilio_call_audio()

        await self.push_frame(frame, direction)

    # End exotel call
    async def _clear_exotel_call_audio(self):
        logger.debug(f"Not trying to clear phone call")
        # data_to_send = {
        #         "event": "clear",
        #         "stream_sid": self.stream_sid
        #     }
        # await self.websocket_client.send_json(json.dumps(data_to_send))
        # logger.debug(f"Exotel phone call audio cleared")
        
    # End twilio call
    async def _clear_twilio_call_audio(self):
        # logger.debug(f"Trying to clear audio phone call")
        data_to_send = {"event": "clear", "streamSid": self.stream_sid}
        await self.websocket_client.send_text(json.dumps(data_to_send))
        # logger.debug(f"Twilio phone call audio cleared")


class IgnorePacketsUntilFirstTTSPacketReceived(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__()
        self.audio_receive_callback = kwargs.get("audio_receive_callback")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # if the frame is AudioRawFrame and _first_tts_audio_received is False then return
        if isinstance(frame, AudioRawFrame) and not self.audio_receive_callback():
            return

        await self.push_frame(frame, direction)


# class which will be similar to transcriptions logger but it will remove space from the text which is in phone number format
class TranscriptionRemoveSpace(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                # remove space from the text which is in phone number format
                text = frame.text
                text = self.clean_phone_number(text)
                logger.debug(f"Audio Transcription: [{text}]")

            await self.push_frame(frame, direction)

        except Exception as e:
            logger.debug(f"Exception {e}")

    def clean_phone_number(self, phone_number):
        if isinstance(phone_number, str):
            # Remove any non-digit characters from the phone number
            cleaned_number = re.sub(r"\D", "", phone_number)
            return cleaned_number
        else:
            return phone_number


# class which will be similar to transcriptions logger but it will translate the text to the given language and then provide the translation text in the process_frame method
class TranslateInput(FrameProcessor):
    def __init__(self, src_lang: str = None, tgt_lang: str = None):
        super().__init__()
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._aggregate_sentences: bool = True
        self._current_sentence: str = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame):
                await self._process_text_frame(frame)
            else:
                await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"Exception {e}")

    async def _process_text_frame(self, frame: TextFrame):
        text: str | None = None
        if not self._aggregate_sentences:
            text = frame.text
        else:
            self._current_sentence += frame.text
            if self._current_sentence.strip().endswith(
                (".", "?", "!", "|")
            ) and not self._current_sentence.strip().endswith(
                ("Mr,", "Mrs.", "Ms.", "Dr.")
            ):
                text = self._current_sentence
                self._current_sentence = ""

        if text:
            # log the text before pushing the translated text frames
            logger.debug(f"Text before generationg translated text: {text}")
            await self._push_translated_text(text)

    # function to translate the text and pass the translated text to the process_frame method
    async def _push_translated_text(self, text: str):
        # log the text before translating
        logger.debug(f"Text before translating: {text}")

        # translation
        # translated_text = await self.translate_text(text, self._src_lang, self._tgt_lang)

        # transliteration
        translated_text = await self.transliterate_text(
            text, self._src_lang, self._tgt_lang
        )

        logger.debug(f"Translated text: [{translated_text}]")

        await self.push_frame(
            TextFrame(text=translated_text), FrameDirection.DOWNSTREAM
        )

    async def translate_text(self, sourceValue, src_lang, tgt_lang):

        try:
            # log the source value, source language and target language
            logger.debug(f"Source value: {sourceValue}")
            logger.debug(f"Source language: {src_lang}")
            logger.debug(f"Target language: {tgt_lang}")

            maskTerms = await self._get_mask_terms(sourceValue)

            # log the mask terms
            logger.debug(f"Mask terms: {maskTerms}")

            # create mapping for mask terms and replace them with %1s, %2s etc
            maskTermsMap = {}
            for index, term in enumerate(maskTerms):
                maskTermsMap[term] = f"%{index + 1}s"

            # replace mask terms with %1s, %2s etc
            for term, replacement in maskTermsMap.items():
                sourceValue = sourceValue.replace(term, replacement)

            data = {"data": [sourceValue], "mask": True}

            headers = {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "REV-APPNAME": "nmt",
                "Content-Type": "application/json;charset=UTF-8",
                "REV-APP-ID": "rev.nlurd",
                "REV-API-KEY": "46a92d8a6b2130b47098e750d9fb046fa37797b9",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://revapi.reverieinc.com/translate",
                    json=data,
                    headers=headers,
                ) as response:
                    response_json = await response.json()

                    # log the response
                    logger.debug(f"Response: {response_json}")

                    translatedValue = response_json["result"][0][0]

                    # log the translated value
                    logger.debug(f"Translated value: {translatedValue}")

                    # replace %1s, %2s etc with mask terms in the translated value
                    for term, replacement in maskTermsMap.items():
                        translatedValue = translatedValue.replace(replacement, term)

                    # log the translated value
                    logger.debug(f"Translated value: {translatedValue}")

                    return translatedValue

            # response = await requests.post('https://revapi.reverieinc.com/translate', json=data, headers=headers)

            # # log the response
            # logger.debug(f"Response: {response.json()}")

            # translatedValue = response.json()["result"][0][0]

            # # log the translated value
            # logger.debug(f"Translated value: {translatedValue}")

            # # replace %1s, %2s etc with mask terms in the translated value
            # for term, replacement in maskTermsMap.items():
            #     translatedValue = translatedValue.replace(replacement, term)

            # # log the translated value
            # logger.debug(f"Translated value: {translatedValue}")

            # return translatedValue

        except Exception as e:
            logger.debug(f"Exception {e}")
            return None

    async def transliterate_text(self, sourceValue, src_lang, tgt_lang):
        try:
            data = {"data": [sourceValue]}

            headers = {
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "REV-APPNAME": "transliteration",
                "Content-Type": "application/json;charset=UTF-8",
                "REV-APP-ID": "rev.nlurd",
                "REV-API-KEY": "46a92d8a6b2130b47098e750d9fb046fa37797b9",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://revapi.reverieinc.com/",
                    json=data,
                    headers=headers,
                ) as response:
                    response_json = await response.json()

                    # # log the response
                    # logger.debug(f"Response: {response_json}")

                    # translatedValue = response_json["result"][0][0]

                    # # log the translated value
                    # logger.debug(f"Translated value: {translatedValue}")

                    # # replace %1s, %2s etc with mask terms in the translated value
                    # for term, replacement in maskTermsMap.items():
                    #     translatedValue = translatedValue.replace(replacement, term)

                    # # log the translated value
                    # logger.debug(f"Translated value: {translatedValue}")

                    # return translatedValue

                    translatedValue = response_json["responseList"][0]["outString"][0]

                    # log the translated value
                    logger.debug(f"Transliterated value: {translatedValue}")

                    return translatedValue

            # response = await requests.post(
            #     "https://revapi.reverieinc.com/", json=data, headers=headers
            # )
            # translatedValue = response.json()["responseList"][0]["outString"][0]

            # # log the translated value
            # logger.debug(f"Transliterated value: {translatedValue}")

            # return translatedValue

        except Exception as e:
            logger.debug(f"Exception in transliterate {e}")
            return None

    async def _get_mask_terms(self, string):
        mask_terms = []
        term = ""
        flag = False
        for char in string:
            if char == "{":
                flag = True
                term += char
            elif char == "}":
                flag = False
                term += char
                mask_terms.append(term)
                term = ""
            elif flag:
                term += char
        return mask_terms


# Create class for LLM response logger
class LLMResposneLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame):
                logger.debug(f"LLM Response: {frame.text}")

            await self.push_frame(frame, direction)

        except Exception as e:
            logger.debug(f"Exception {e}")


class ConversationEndManager(FrameProcessor):
    def __init__(self, websocket_client: WebSocket, **kwargs):
        super().__init__()
        self.keywords = [
            "goodbye",
            "bye",
            "exit",
            "quit",
            "have a great day",
            "end flow",
            "/quit",
            "automatic end display",
            "this is the final display",
            "बाई बाई",
            "बाई",
            "अलविदा",
            "थैंक यू",
            "रखिए कॉल",
        ]
        self._aggregate_sentences: bool = True
        self._current_sentence: str = ""
        self.call_sid = kwargs.get("call_sid")
        self.websocket_client = websocket_client
        self.stream_sid = kwargs.get("stream_sid")
        self._save_conversation_callback = kwargs.get("save_conversation_callback")
        self._summarise_conversation_callback = kwargs.get(
            "summarise_conversation_callback"
        )
        self._call_provider = kwargs.get("call_provider")

    async def _end_exotel_call(self):
        logger.debug(f"Trying to end the phone call")
        data_to_send = {"event": "stop", "stream_sid": self.stream_sid}
        await self.websocket_client.send_json(json.dumps(data_to_send))
        logger.debug(f"Phone call ended")

        await self._save_conversation_callback()
        await self._summarise_conversation_callback()

    # End twilio call if any exception occurs
    async def _end_twilio_call(self):
        logger.debug(f"Conversation ends")

        # get the call_sid that is passed in the __init__ method
        call_sid = self.call_sid

        # print the call_sid
        logger.debug(f"call_sid: {call_sid}")

        if call_sid is None:
            raise Exception("call_sid is missing in the __init__ method")

        try:
            # get the call sid from the call_sid
            twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")

            # create a twilio client
            twilioclient = Client(twilio_account_sid, twilio_auth_token)

            # log the call_sid
            logger.debug(f"call_sid: {call_sid}")

            # update the call status to completed
            call = twilioclient.calls(call_sid).update(status="completed")

            # add debug log that call successfully closed
            logger.debug(f"Call successfully closed. call_sid: {call_sid}")

            await self._save_conversation_callback()
            await self._summarise_conversation_callback()

        except Exception as e:
            raise Exception(f"Failed to forward call: {str(e)}")

    async def _process_text_frame(self, frame: TextFrame):
        text: str | None = None
        if not self._aggregate_sentences:
            text = frame.text
        else:
            self._current_sentence += frame.text
            if self._current_sentence.strip().endswith(
                (".", "?", "!")
            ) and not self._current_sentence.strip().endswith(
                ("Mr,", "Mrs.", "Ms.", "Dr.")
            ):
                text = self._current_sentence
                self._current_sentence = ""

        if text:
            if any(keyword in text.lower() for keyword in self.keywords):
                await asyncio.sleep(5)  # Wait for 5 seconds
                if self._call_provider == "twilio":
                    await self._end_twilio_call()
                elif self._call_provider == "exotel":
                    await self._end_exotel_call()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame):
                if frame.text == "End Flow":
                    logger.debug("I am here, waiting...")
                await self._process_text_frame(frame)

            await self.push_frame(frame, direction)

        except Exception as e:
            logger.debug(f"Exception {e}")


class WelcomeMessageLLMResponseProcessor(FrameProcessor):
    def __init__(self, welcome_message):
        super().__init__()
        self.text_sent = False
        self._welcome_message = welcome_message

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if not self.text_sent and self._welcome_message:
                await self.push_frame(TextFrame(self._welcome_message))
                self.text_sent = True

            await self.push_frame(frame, direction)

        except Exception as e:
            logger.debug(f"Exception {e}")


class FirstTTSAudioEndProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        callback: Callable[["FirstTTSAudioEndProcessor"], Awaitable[None]],
    ):
        super().__init__()
        self._callback = callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)

            # logger.debug(f"Frame inside IgnoreAudioUntilTTS: {frame}")
            if isinstance(frame, TTSStoppedFrame):
                await self._callback(self)

            # continue pushing the frames
            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"Exception {e}")


class AudioVolumeTimer(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.last_transition_ts = 0
        self._prev_volume = -80
        self._speech_volume_threshold = -50

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            volume = self.calculate_volume(frame)
            # print(f"Audio volume: {volume:.2f} dB")
            if (
                volume >= self._speech_volume_threshold
                and self._prev_volume < self._speech_volume_threshold
            ):
                # logger.debug("transition above speech volume threshold")
                self.last_transition_ts = time.time()
            elif (
                volume < self._speech_volume_threshold
                and self._prev_volume >= self._speech_volume_threshold
            ):
                # logger.debug("transition below non-speech volume threshold")
                self.last_transition_ts = time.time()
            self._prev_volume = volume

        await self.push_frame(frame, direction)

    def calculate_volume(self, frame: AudioRawFrame) -> float:
        if frame.num_channels != 1:
            raise ValueError(f"Expected 1 channel, got {frame.num_channels}")

        # Unpack audio data into 16-bit integers
        fmt = f"{len(frame.audio) // 2}h"
        audio_samples = struct.unpack(fmt, frame.audio)

        # Calculate RMS
        sum_squares = sum(sample**2 for sample in audio_samples)
        rms = math.sqrt(sum_squares / len(audio_samples))

        # Convert RMS to decibels (dB)
        # Reference: maximum value for 16-bit audio is 32767
        if rms > 0:
            db = 20 * math.log10(rms / 32767)
        else:
            db = -96  # Minimum value (almost silent)

        return db
