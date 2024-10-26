
import asyncio
import os
from typing import List, AsyncGenerator

# from litellm import speech
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    VisionImageRawFrame,
    LLMMessagesFrame,
    LLMFullResponseStartFrame,
    LLMResponseStartFrame,
    LLMResponseEndFrame,
    LLMFullResponseEndFrame,
    AudioRawFrame,
    ErrorFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService, TTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame

from loguru import logger

try:
    import google.generativeai as gai
    import google.ai.generativelanguage as glm
    from google.cloud import texttospeech
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set `GOOGLE_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class GoogleLLMService(LLMService):
    """This class implements inference with Google's AI models

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

    def __init__(self, *, api_key: str, model: str = "gemini-1.5-flash-latest", **kwargs):
        super().__init__(**kwargs)
        gai.configure(api_key=api_key)
        self._client = gai.GenerativeModel(model)

    def can_generate_metrics(self) -> bool:
        return True

    def _get_messages_from_openai_context(
            self, context: OpenAILLMContext) -> List[glm.Content]:
        openai_messages = context.get_messages()
        google_messages = []

        for message in openai_messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                role = "user"
            elif role == "assistant":
                role = "model"

            parts = [glm.Part(text=content)]
            if "mime_type" in message:
                parts.append(
                    glm.Part(inline_data=glm.Blob(
                        mime_type=message["mime_type"],
                        data=message["data"].getvalue()
                    )))
            google_messages.append({"role": role, "parts": parts})

        return google_messages

    async def _async_generator_wrapper(self, sync_generator):
        for item in sync_generator:
            yield item
            await asyncio.sleep(0)

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            logger.debug(f"Generating chat: {context.get_messages_json()}")

            messages = self._get_messages_from_openai_context(context)

            await self.start_ttfb_metrics()

            response = self._client.generate_content(messages, stream=True)

            await self.stop_ttfb_metrics()

            async for chunk in self._async_generator_wrapper(response):
                try:
                    text = chunk.text
                    await self.push_frame(LLMResponseStartFrame())
                    await self.push_frame(TextFrame(text))
                    await self.push_frame(LLMResponseEndFrame())
                except Exception as e:
                    # Google LLMs seem to flag safety issues a lot!
                    if chunk.candidates[0].finish_reason == 3:
                        logger.debug(
                            f"LLM refused to generate content for safety reasons - {messages}.")
                    else:
                        logger.exception(f"{self} error: {e}")

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None

        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)


class GoogleTTSService(TTSService):
    def __init__(
        self,
        *,
        client: texttospeech.TextToSpeechAsyncClient,
        voice_name: str = "en-US-Standard-C",
        language_code: str = "en-US",
        gender: str = "FEMALE",  # Add gender parameter with default value
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._voice_name = voice_name
        self._language_code = language_code
        self._gender = gender  # Store the gender parameter

    def can_generate_metrics(self) -> bool:
        return True

    def generate_file_path(self, text: str) -> str:
        cache_dir = "cache/tts_cache"
        os.makedirs(cache_dir, exist_ok=True)
    
        unique_id = str(hash(text))
        file_path = os.path.join(cache_dir, f"{unique_id}.wav")
        
        return file_path

    def save_audio_data(self, text, audio_content):
        try:
            file_path = self.generate_file_path(f"{text}_{self._voice_name}")
            with open(file_path, "wb") as f:
                f.write(audio_content)
        except Exception as e:
            logger.error(f"Error saving audio data: {e}")

    def get_audio_data(self, text):
        try:
            file_path = self.generate_file_path(f"{text}_{self._voice_name}")
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    with open(file_path, "rb") as f:
                        audio_data = f.read()
                    logger.debug(f"Audio data fetched from cache: {file_path}")
                    return audio_data
            return None
        except Exception as e:
            logger.error(f"Error fetching audio data: {e}")
            return None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        audio_data = self.get_audio_data(text)
        
        if audio_data:
            logger.debug(f"Audio data found in cache for text: {text}")
            await self.start_ttfb_metrics()
            for i in range(0, len(audio_data), 16000):
                await self.stop_ttfb_metrics()
                chunk = audio_data[i:i+16000]
                frame = AudioRawFrame(chunk, 16000, 1)
                yield frame
        else:
            logger.debug(f"Generating audio data for text: {text}")
            try:
                input_text = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code=self._language_code,
                    name=self._voice_name,
                    ssml_gender=self._gender  # Set the gender parameter
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000
                )

                await self.start_ttfb_metrics()

                response = await self._client.synthesize_speech(
                    input=input_text,
                    voice=voice,
                    audio_config=audio_config
                )

                await self.stop_ttfb_metrics()

                audio_content = response.audio_content
                self.save_audio_data(text, audio_content)

                for i in range(0, len(audio_content), 16000):
                    chunk = audio_content[i:i+16000]
                    frame = AudioRawFrame(chunk, 16000, 1)
                    yield frame

            except Exception as e:
                logger.error(f"Error generating audio: {e}")
                yield ErrorFrame(f"Error generating audio: {e}")

    @classmethod
    async def create(cls, **kwargs):
        client = texttospeech.TextToSpeechAsyncClient()
        return cls(client=client, **kwargs)


class GoogleSTTService:
    def __init__(self, *, api_key: str, language_code: str = "en-US"):
        self._api_key = api_key
        self._language_code = language_code
        self._client = None

    async def initialize(self):
        from google.cloud import speech_v1p1beta1 as speech
        self._client = speech.SpeechAsyncClient()

    async def transcribe(self, audio_content: bytes) -> str:
        if not self._client:
            await self.initialize()

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self._language_code,
        )

        response = await self._client.recognize(config=config, audio=audio)

        transcription = ""
        for result in response.results:
            transcription += result.alternatives[0].transcript

        return transcription

    async def stream_transcribe(self, audio_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        if not self._client:
            await self.initialize()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self._language_code,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

        async def request_generator():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            async for content in audio_generator:
                yield speech.StreamingRecognizeRequest(audio_content=content)

        responses = self._client.streaming_recognize(request_generator())

        async for response in responses:
            for result in response.results:
                if result.is_final:
                    yield result.alternatives[0].transcript

    @classmethod
    async def create(cls, **kwargs):
        instance = cls(**kwargs)
        await instance.initialize()
        return instance

