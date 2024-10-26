import json

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame,
    VisionImageRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from loguru import logger
import random
import string
import re
import asyncio

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Reverie Chatter, you need to `pip install websockets`. Also, make sure you have the necessary configuration in place."
    )
    raise Exception(f"Missing module: {e}")


class ReverieChatterLLMService(LLMService):
    """This class implements a language model service for Reverie Chatter.
    This service is responsible for generating responses using the Reverie Chatter language model.
    It translates internally from OpenAILLMContext to the messages format expected by the Reverie Chatter AI model.
    The OpenAILLMContext is used as a common format for communication between different language models.
    The ReverieChatterLLMService handles the processing of incoming frames, generates Reverie Chatter responses,
    and pushes the response frames to the frame processor for further processing.
    """

    def __init__(
        self,
        url: str = "wss://sansadhak-response.reverieinc.com/api/chatter",
        bot_details: dict = {},
        set_current_language: callable = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._conversation_id = ""
        self._url = url
        self._call_sid = self._generate_random_id(16)
        self._bot_details = bot_details
        self._set_current_language = set_current_language
        self._websocket = None

    async def start(self, frame: StartFrame):
        # get the api details from the bot details
        api_details = self._bot_details.get("api_details", {})

        # get the selectLanguage
        select_language = self._bot_details.get("selectLanguage", "")

        # log the select_language
        logger.debug(f"Select Language: {select_language}")

        # if select_language is False, then connect with the websocket otherwise send the preferred language message
        if select_language == "False" or select_language == False:
            await self._connect()
        else:
            await self._send_preferred_language_message(
                api_details.get("available_languages", [])
            )

    # logic to connect to the rev_chatter websocket
    async def _connect(self, language_code=None):
        self._websocket = await websockets.connect(self._url)
        self._receive_task = self.get_event_loop().create_task(
            self._receive_task_handler()
        )
        await self._setup_rev_chatter(language_code)

        # set the current language
        if self._set_current_language and language_code is not None:
            self._set_current_language(language_code)

    # function to return the language based on the language code
    def _get_language(self, language_code):
        language_map = {
            "en": "English",
            "hi": "Hindi",
            "bn": "Bengali",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "mr": "Marathi",
            "or": "Odia",
            "pa": "Punjabi",
            "ta": "Tamil",
            "te": "Telugu",
            "ur": "Urdu",
        }

        return language_map.get(language_code, "")

    # function to return the language code based on the language
    def _get_language_code(self, language):
        
        # strip the language and convert to lowercase and also remove . character
        language = language.strip().lower().replace(".", "")
        
        # log the language
        logger.debug(f"Language: {language}")

        language_map = {
            "english": "en",
            "hindi": "hi",
            "bengali": "bn",
            "gujarati": "gu",
            "kannada": "kn",
            "malayalam": "ml",
            "marathi": "mr",
            "odia": "or",
            "oriya": "or",
            "punjabi": "pa",
            "tamil": "ta",
            "telugu": "te",
            "urdu": "ur",
            "bangla": "bn",
            "assamese": "as",
            "arabic": "ar",
            "oh yeah": "or",
            "canada": "kn",
        }

        # check for exact matching
        if language in language_map:
            return language_map[language]

        # check for partial matching
        for key in language_map:
            if key in language:
                return language_map[key]

        return ""

    async def _setup_rev_chatter(self, language_code=None):
        # get the api details from the bot details
        api_details = self._bot_details.get("api_details", {})

        # get the available_languages
        available_languages = api_details.get("available_languages", [])

        # log the api details
        logger.debug(f"API Details: {api_details}")

        # log the language_code
        logger.debug(f"Language Code: {language_code}")

        # if language_code is present in the available_languages, then add the language code to the api details
        if language_code is not None and language_code in available_languages:
            api_details["src_language"] = language_code

        # eastman auto bot
        configuration = {
            "event": "CALL_STARTED",
            "callSid": self._call_sid,
            "type": "chatty",
            "user_id": "",
            "center_id": "",
            "api_details": api_details,
        }

        # log the configuration
        logger.debug(f"Configuration: {configuration}")

        await self._websocket.send(json.dumps(configuration))

    # function to generate random id
    def _generate_random_id(self, length: int) -> str:
        letters_and_digits = string.ascii_letters + string.digits
        return "".join(random.choice(letters_and_digits) for _ in range(length))

    # send text data to the rev_chatter websocket
    async def _send_text(self, text: str):
        # log the text
        logger.debug(f"Text: {text}")

        message = {
            "event": "DATA",
            "callSid": self._call_sid,
            "data": text,
            "stt_tag": "null",
            "center_id": "",
            "contact": "",
        }

        await self._websocket.send(json.dumps(message))

    def _clean_phone_number(self, phone_number):
        if re.match(
            r"^\d{3}[-\s]?\d{3}[-\s]?\d{4}$",
            phone_number.replace(" ", "").replace("-", "").replace(".", ""),
        ):
            # Remove any non-digit characters from the phone number
            cleaned_number = re.sub(r"\D", "", phone_number)
            return cleaned_number
        else:
            return phone_number

    async def _handle_language_select(self, language):
        
        # log the language
        logger.debug(f"Language: {language}")
        
        # get the language code based on the language
        language_code = self._get_language_code(language)

        # log the language code
        logger.debug(f"Language Code: {language_code}")

        # connect with the rev_chatter websocket with language code
        await self._connect(language_code)

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            logger.debug(f"Generating chat: {context.get_messages_json()}")

            user_messages = context.get_messages()

            # if the user messages are empty, then return
            if not user_messages and user_messages == []:
                return

            user_message = context.get_messages()[-1]

            # get the content from the user message
            user_message_content = user_message["content"]

            # filter the phone number from the user message content
            updated_message = self._clean_phone_number(user_message_content)

            # log the messages
            logger.debug(f"User Input: [{updated_message}]")

            # if self._websocket is present then send the text data to the rev_chatter websocket, otherwise connect with chatter websocket
            if not self._websocket:
                await self._handle_language_select(updated_message)
            else:
                await self.start_ttfb_metrics()

                # log the user message content
                logger.debug(f"User Message Content: {updated_message}")

                # send text data to the rev_chatter websocket
                await self._send_text(updated_message)

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

    # function to send message "Please choose your preferred language" as the first message
    async def _send_preferred_language_message(self, available_languages):
        # log the api details
        logger.debug(f"available languages: {available_languages}")

        # set the current language
        if self._set_current_language:
            self._set_current_language("choice")

        # send the message "Please choose your preferred language"
        await self.push_frame(LLMResponseStartFrame())
        await self.push_frame(TextFrame("Please choose your preferred language."))
        await self.push_frame(LLMResponseEndFrame())

        # pass the available languages also to the next frame
        await self.push_frame(LLMResponseStartFrame())
        for language_code in available_languages:
            languge = self._get_language(language_code)
            await self.push_frame(TextFrame(languge))
        await self.push_frame(LLMResponseEndFrame())

    async def _receive_task_handler(self):
        async for message in self._websocket:
            chatter_message = json.loads(message)

            # log the chatter message
            logger.debug(f"Chatter Message: {chatter_message}")

            response = chatter_message.get("response")

            bot_selected_language = chatter_message.get("language")

            # log the bot_selected_language
            # logger.debug(f"Bot Selected Language: {bot_selected_language}")

            # if bot_selected_language is present, then set the current language
            if self._set_current_language and bot_selected_language is not None:
                self._set_current_language(bot_selected_language)

            response_utterance = ""
            response_options = None
            # convert the response to a json object
            try:
                response_json = json.loads(response)

                # check for type of response
                if response_json.get("type", "") == "delay":
                    
                    # log that the response is of type delay
                    logger.debug(f"Response is of type delay")
                    
                    # if delay sleep for the delay durations
                    # await asyncio.sleep(response_json.get("delay_duration", 0))
                    continue

                # get the utterance from the response json
                response_utterance = response_json.get("utterance")

                # get the options from the response json
                response_options = response_json.get("options", {})

                # get the variable_keyname from the response json
                variable_keyname = chatter_message.get("variable_keyname", "")

            except Exception as e:
                response_utterance = response

            if response_utterance in ["This is the final display.", "Automatic end display.", "End Flow"]:
                await self._send_text("/quit")
                continue

            # if response utterance is $[DELAY-1] then return
            if response_utterance == "$[DELAY-1]":
                logger.debug(f"Response Utterance is $[DELAY-1]")
                continue
            
            # elif response_utterance == "Automatic end display":
            #     logger.debug(f"Ignoring automatic end display")
            #     continue
            else:
                # log the response utterance
                logger.debug(
                    f"Passing the response_utterance from the llm: {response_utterance}"
                )

                updated_response_utterance = response_utterance

                if not updated_response_utterance.strip().endswith((".", "?", "!", "|")):
                    updated_response_utterance = response_utterance + "."

                # if response options are present, then append the response options to the response utterance and send the response utterance
                if response_options:
                    # loop through the options and append sentence ending symbols
                    for key, value in response_options.items():
                        # for language options we are appending the related sentence end character
                        if variable_keyname == "active_language":
                            if value == "English":
                                updated_response_utterance += f" {value}."
                            else:
                                updated_response_utterance += f" {value}|"
                        # otherwise we are appending . character
                        else:
                            updated_response_utterance += f" {value}."
        
                # log the updated response utterance
                logger.debug(
                    f"Updated Response Utterance: {updated_response_utterance}"
                )

                # if variable_keyname equals "active_language" then set the current language to choice.
                # We are doing this because the language selection is done by the user and we need to set the current language to choice
                if variable_keyname == "active_language":
                    if self._set_current_language:
                        self._set_current_language("choice")

                await self.push_frame(LLMResponseStartFrame())
                await self.push_frame(TextFrame(updated_response_utterance))
                await self.push_frame(LLMResponseEndFrame())
