
import aiohttp
import asyncio
import io
import time
import json
import re

from PIL import Image
from typing import AsyncGenerator

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
    URLImageRawFrame)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AsyncAIService, TTSService, ImageGenService
from pipecat.services.openai import BaseOpenAILLMService

from loguru import logger

# See .env.example for Azure configuration needed
try:
    from openai import AsyncAzureOpenAI
    from azure.cognitiveservices.speech import (
        SpeechConfig,
        SpeechRecognizer,
        SpeechSynthesizer,
        ResultReason,
        CancellationReason,
        PhraseListGrammar
    )
    from azure.cognitiveservices.speech.audio import AudioStreamFormat, PushAudioInputStream
    from azure.cognitiveservices.speech.dialog import AudioConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Azure, you need to `pip install pipecat-ai[azure]`. Also, set `AZURE_SPEECH_API_KEY` and `AZURE_SPEECH_REGION` environment variables.")
    raise Exception(f"Missing module: {e}")


class AzureLLMService(BaseOpenAILLMService):
    def __init__(
            self,
            *,
            api_key: str,
            endpoint: str,
            model: str,
            api_version: str = "2023-12-01-preview"):
        # Initialize variables before calling parent __init__() because that
        # will call create_client() and we need those values there.
        self._endpoint = endpoint
        self._api_version = api_version
        super().__init__(api_key=api_key, model=model)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        
        # log the api key, azure endpoint and api version
        logger.debug(f"API Key: {api_key}")
        logger.debug(f"Azure Endpoint: {self._endpoint}")
        logger.debug(f"API Version: {self._api_version}")
        
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )


class AzureTTSService(TTSService):
    def __init__(self, *, api_key: str, region: str, voice="en-US-SaraNeural", **kwargs):
        super().__init__(**kwargs)

        speech_config = SpeechConfig(subscription=api_key, region=region)
        self._speech_synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        self._voice = voice

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()

        ssml = (
            "<speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{self._voice}'>"
            "<mstts:silence type='Sentenceboundary' value='20ms' />"
            "<mstts:express-as style='lyrical' styledegree='2' role='SeniorFemale'>"
            "<prosody rate='1.05'>"
            f"{text}"
            "</prosody></mstts:express-as></voice></speak> ")

        result = await asyncio.to_thread(self._speech_synthesizer.speak_ssml, (ssml))
        
        # log the result
        logger.debug(f"Result: {result}")

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            await self.stop_ttfb_metrics()
            # Azure always sends a 44-byte header. Strip it off.
            yield AudioRawFrame(audio=result.audio_data[44:], sample_rate=16000, num_channels=1)
        elif result.reason == ResultReason.Canceled:
            
            logger.warning(f"Speech synthesis canceled: {result.cancellation_details.reason}")
            
            cancellation_details = result.cancellation_details
            logger.warning(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                logger.error(f"{self} error: {cancellation_details.error_details}")


class AzureSTTService(AsyncAIService):
    def __init__(
            self,
            *,
            api_key: str,
            region: str,
            language="en-US",
            sample_rate=16000,
            channels=1,
            **kwargs):
        super().__init__(**kwargs)
        
        # get available_languages from the arguments
        available_languages = kwargs.get("available_languages", [])
        
        # log the available languages
        logger.debug(f"Available languages in azure stt: {available_languages}")
        
        # log the language
        logger.debug(f"Language in azure stt: {language}")
        
        speech_config = SpeechConfig(subscription=api_key, region=region)
        speech_config.speech_recognition_language = language
        
        stream_format = AudioStreamFormat(samples_per_second=sample_rate, channels=channels)
        self._audio_stream = PushAudioInputStream(stream_format)

        audio_config = AudioConfig(stream=self._audio_stream)
        self._speech_recognizer = SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config)
        self._speech_recognizer.recognized.connect(self._on_handle_recognized)
        
        # if the available languages are not empty then get the language from the available languages and add to the speech grammar
        if len(available_languages) > 1:
            phrase_list_grammar = PhraseListGrammar(recognizer=self._speech_recognizer)
            for language_code in available_languages:
                language = self._get_language_name(language_code)
                phrase_list_grammar.addPhrase(language)
                
            # Add phrase list grammar to speech config
            self._speech_recognizer.dynamic_phrase_list_grammar = phrase_list_grammar
            
            # log the phrase list grammar
            logger.debug(f"Phrase list grammar: {phrase_list_grammar}") 


    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            self._audio_stream.write(frame.audio)
        else:
            await self._push_queue.put((frame, direction))

    async def start(self, frame: StartFrame):
        self._speech_recognizer.start_continuous_recognition_async()

    async def stop(self, frame: EndFrame):
        self._speech_recognizer.stop_continuous_recognition_async()
        self._audio_stream.close()

    async def cancel(self, frame: CancelFrame):
        self._speech_recognizer.stop_continuous_recognition_async()
        self._audio_stream.close()

    def _on_handle_recognized(self, event):
        if event.result.reason == ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            
            transcript = event.result.text
            
            # updated_transcript = await self._process_punctuation(transcript)
            _transcript = self._process_numbers(transcript)
            
            # remove spaces and commas between numbers
            _updated_number = self._remove_spaces_followed_by_number(_transcript)
                            
            # _entity_transcript = await self.process_entities(_updated_number, self._language)

            # log the updated transcript
            # logger.debug(f"Updated transcript: {_entity_transcript}")
            
            frame = TranscriptionFrame(_updated_number, "", int(time.time_ns() / 1000000))
            asyncio.run_coroutine_threadsafe(self.queue_frame(frame), self.get_event_loop())


    # allLanguages = ["hi", "en", "bn", "as", "kn", "ml", "mr", "or", "ta", "te", "pa", "gu"]
    # create a function to get the language name from the available languages
    def _get_language_name(self, language_code):
        # create a dictionary of language code and language name
        language_dict = {
            "hi": "hindi",
            "en": "english",
            "bn": "bengali",
            "as": "assamese",
            "kn": "kannada",
            "ml": "malayalam",
            "mr": "marathi",
            "or": "oriya",
            "ta": "tamil",
            "te": "telugu",
            "pa": "punjabi",
            "gu": "gujarati",
            "ar": "arabic",
        }
        
        # get the language name from the language code
        language_name = language_dict.get(language_code)
        
        # return the language name
        return language_name
    

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

    def _process_numbers(self, transcript: str):
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


class AzureImageGenServiceREST(ImageGenService):

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        image_size: str,
        api_key: str,
        endpoint: str,
        model: str,
        api_version="2023-06-01-preview",
    ):
        super().__init__()

        self._api_key = api_key
        self._azure_endpoint = endpoint
        self._api_version = api_version
        self._model = model
        self._aiohttp_session = aiohttp_session
        self._image_size = image_size

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        url = f"{self._azure_endpoint}openai/images/generations:submit?api-version={self._api_version}"

        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"}

        body = {
            # Enter your prompt text here
            "prompt": prompt,
            "size": self._image_size,
            "n": 1,
        }

        async with self._aiohttp_session.post(url, headers=headers, json=body) as submission:
            # We never get past this line, because this header isn't
            # defined on a 429 response, but something is eating our
            # exceptions!
            operation_location = submission.headers["operation-location"]
            status = ""
            attempts_left = 120
            json_response = None
            while status != "succeeded":
                attempts_left -= 1
                if attempts_left == 0:
                    logger.error(f"{self} error: image generation timed out")
                    yield ErrorFrame("Image generation timed out")
                    return

                await asyncio.sleep(1)

                response = await self._aiohttp_session.get(operation_location, headers=headers)

                json_response = await response.json()
                status = json_response["status"]

            image_url = json_response["result"]["data"][0]["url"] if json_response else None
            if not image_url:
                logger.error(f"{self} error: image generation failed")
                yield ErrorFrame("Image generation failed")
                return

            # Load the image from the url
            async with self._aiohttp_session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                frame = URLImageRawFrame(
                    url=image_url,
                    image=image.tobytes(),
                    size=image.size,
                    format=image.format)
                yield frame
