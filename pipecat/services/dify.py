import asyncio
import json
import requests
import aiohttp
import time
from typing import List
from requests import RequestException
from pipecat.services.nmt import NMTService
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame,
    VisionImageRawFrame,
    ErrorFrame,
    AudioRawFrame,
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
from typing import AsyncGenerator
import http.client
import json

class DifyLLMService(LLMService):
    """This class implements a language model service for Dify.

    This service is responsible for generating responses using the Dify language model.
    It translates internally from OpenAILLMContext to the messages format expected by the Dify AI model.
    The OpenAILLMContext is used as a common format for communication between different language models.
    The DifyLLMService handles the processing of incoming frames, generates Dify responses,
    and pushes the response frames to the frame processor for further processing.
    """

    def __init__(
        self,
        aiohttp_session: aiohttp.ClientSession,
        api_key: str,
        save_bot_context,
        tgt_lan: str,
        nmt_flag: str,
        nmt_provider: str,
        url: str = "https://conversation-app.reverieinc.com/v1",
        frame: str="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._conversation_id = ""
        self._api_key = api_key
        self._tgt_lan=tgt_lan
        self._src_lan="en"
        self._url = url
        self._frame=frame
        self._processed_text=""
        self._user_id = self._generate_random_id(8)
        self._save_bot_context = save_bot_context
        self._nmt_flag=nmt_flag
        self._nmt_provider = nmt_provider

    # function to generate random id
    def _generate_random_id(self, length: int) -> str:
        letters_and_digits = string.ascii_letters + string.digits
        return "".join(random.choice(letters_and_digits) for _ in range(length))

    # function to set the conversation id
    def set_conversation_id(self, conversation_id: str):
        self._conversation_id = conversation_id

    def can_generate_metrics(self) -> bool:
        return True
   

    async def _stream_dify_response(self, text: str):
        url = f"{self._url}/chat-messages"

        payload = json.dumps(
            {
                "inputs": {},
                "query": text,
                "response_mode": "streaming",
                "conversation_id": self._conversation_id,
                "user": self._user_id,
                "files": [
                    # {
                    #     "type": "image",
                    #     "transfer_method": "remote_url",
                    #     "url": "https://cloud.dify.ai/logo/logo-site.png",
                    # }
                ],
            }
        )
        headers = {
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
        }
        
        try:
            logger.debug(f"URL: {url}")
            logger.debug(f"Payload: {payload}")
            
            with requests.post(url, headers=headers, data=payload, stream=True) as response:
                for line in response.iter_lines():
                    if line.startswith(b'data:'):
                        data = line[len(b'data:'):].strip()
                        # Convert bytes to string and then load as JSON
                        data_json = json.loads(data.decode('utf-8'))
                        # Use the data from the first event or any event as needed
                        # logger.debug(data_json)
                        
                        # save the conversation id so that we can use it in the next request
                        if not self._conversation_id:
                            conversation_id = data_json.get("conversation_id")
                            # log the conversation id
                            logger.debug(f"Conversation ID: {conversation_id}")
                            self.set_conversation_id(conversation_id)
                        
                        # get the answer from the response
                        dify_answer_text = data_json.get("answer")
                        # logger.debug(f"Dify LLM Response: {dify_answer_text}")
                        
                        if self._nmt_flag == True:

                            if dify_answer_text is not None:
                                self._frame +=dify_answer_text
                                
                            # logger.debug(f"frame: {self._frame}")
                            
                            # consolidate the data
                            if self._frame.strip().endswith(
                                (".", "?", "!", "|","।")) and not self._frame.strip().endswith(
                                ("Mr,", "Mrs.", "Ms.", "Dr.")):
                                text = self._frame
                                text = text.replace("*","")
                                logger.debug(f"consolidated: {text}")
                                translator = NMTService(text,self._tgt_lan,self._nmt_provider)
                                # processed_text=await NMTService.reverie_nmt(text,self._tgt_lan)
                                processed_text = await translator.translate()
                                # processed_text=await self.reverie_nmt(text,self._tgt_lan,self._src_lan)
                                # processed_text=await self.google_nmt(text,self._tgt_lan,self._src_lan)
                                # self._processed_text = processed_text[0]
                                self._processed_text = processed_text
                                logger.debug(f"processed_text: {self._processed_text}")
                                self._frame = ""
                        else:
                            if dify_answer_text is not None: 
                                dify_answer_text = dify_answer_text.replace("*","")
                            self._processed_text = dify_answer_text
                            # self._processed_text = self._processed_text.replace("*","")

                        if self._processed_text: # if the dify_answer_text was None it was throwing errors
                            await self.push_frame(LLMResponseStartFrame())
                            await self.push_frame(TextFrame(self._processed_text))
                            await self.push_frame(LLMResponseEndFrame())
                            self._processed_text=""
                        
        except Exception as e:
            logger.error(f"Error in API call: {e}")


    async def _generate_dify_response(self, text: str):
        url = f"{self._url}/chat-messages"

        payload = json.dumps(
            {
                "inputs": {},
                "query": text,
                "response_mode": "streaming",
                "conversation_id": self._conversation_id,
                "user": self._user_id,
                "files": [
                    {
                        "type": "image",
                        "transfer_method": "remote_url",
                        "url": "https://cloud.dify.ai/logo/logo-site.png",
                    }
                ],
            }
        )
        headers = {
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.request("POST", url, headers=headers, data=payload, stream=True)
            res = response.text.split("data:")

            # print the response
            logger.debug(f"Response from Dify: {res}")

            await self.stop_ttfb_metrics()

            index = 0

            while index < len(res):
                if res[index].strip() != "":
                    message_body = json.loads(res[index].strip())

                    # save the conversation id so that we can use it in the next request
                    if not self._conversation_id:
                        conversation_id = message_body.get("conversation_id")
                        self.set_conversation_id(conversation_id)

                    if message_body["event"] == "message_end":
                        break

                    # message for chatbot | agent_message for agent
                    if message_body["event"] in ["message", "agent_message"]:
                        dify_answer_text = message_body["answer"]
                        await self.push_frame(LLMResponseStartFrame())
                        await self.push_frame(TextFrame(dify_answer_text))
                        await self.push_frame(LLMResponseEndFrame())
                index = index + 1
        except Exception as e:
            logger.error(f"Error in api call dify:{e}")
            return {}

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            logger.debug(f"Generating chat: {context.get_messages_json()}")
            

            # get the last object from the context
            user_message = context.get_messages()[-1]
            
            # get context messages 
            context_messages = context.get_messages()
            self._save_bot_context(context_messages)

            # get the content from the user message
            user_message_content = user_message["content"]

            # log the messages
            logger.debug(f"User Input: [{user_message_content}]")

            await self.start_ttfb_metrics()

            # call stream dify response
            await self._stream_dify_response(user_message_content)

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
