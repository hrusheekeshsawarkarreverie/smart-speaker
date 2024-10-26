# #app.py

# import asyncio
# import aiohttp
# import os

# from pipecat.frames.frames import EndFrame, TextFrame
# from pipecat.pipeline.pipeline import Pipeline
# from pipecat.pipeline.task import PipelineTask
# from pipecat.pipeline.runner import PipelineRunner
# from pipecat.services.elevenlabs import ElevenLabsTTSService
# from pipecat.transports.services.daily import DailyParams, DailyTransport
# from pipecat.vad.silero import SileroVADAnalyzer
# from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

# from dotenv import load_dotenv
# load_dotenv()


# async def main():
# #   async with aiohttp.ClientSession() as session:
# #     # Use Daily as a real-time media transport (WebRTC)
# #     transport = DailyTransport(
# #       room_url="https://reverie.daily.co/lsR9SZ9JLnHgeG8DrtBA",
# #       token="87aab209f616bd6eb9e7ec1a1250d70aaa3c07ca548b4ca85d2d1db16f904ba2",
# #       bot_name="Bot Name",
# #       params=DailyParams(audio_out_enabled=True))

#     async with aiohttp.ClientSession() as session:
#         # (room_url, token) = await configure(session)
#         # url =  os.getenv("DAILY_SAMPLE_ROOM_URL")
#         # key =  os.getenv("DAILY_API_KEY")
#         url ="https://reverie.daily.co/reverie_demo"
#         key="87aab209f616bd6eb9e7ec1a1250d70aaa3c07ca548b4ca85d2d1db16f904ba2"
#         if not url:
#             raise Exception(
#                 "No Daily room specified. use the -u/--url option from the command line, or set DAILY_SAMPLE_ROOM_URL in your environment to specify a Daily room URL.")

#         if not key:
#             raise Exception("No Daily API key specified. use the -k/--apikey option from the command line, or set DAILY_API_KEY in your environment to specify a Daily API key, available from https://dashboard.daily.co/developers.")

#         daily_rest_helper = DailyRESTHelper(
#             daily_api_key=key,
#             daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
#             # aiohttp_session=aiohttp_session
#         )

#         # Create a meeting token for the given room with an expiration 1 hour in
#         # the future.
#         expiry_time: float = 60 * 60
#         room_url="https://reverie.daily.co/lsR9SZ9JLnHgeG8DrtBA",

#         # token = await daily_rest_helper.get_token(url, expiry_time)
#         token = daily_rest_helper.get_token(url, expiry_time)
#         transport = DailyTransport(
#             room_url,
#             token,
#             "Chatbot",
#             DailyParams(
#                 audio_out_enabled=True,
#                 camera_out_enabled=True,
#                 camera_out_width=1024,
#                 camera_out_height=576,
#                 vad_enabled=True,
#                 vad_analyzer=SileroVADAnalyzer(),
#                 transcription_enabled=True,
#                 #
#                 # Spanish
#                 #
#                 # transcription_settings=DailyTranscriptionSettings(
#                 #     language="es",
#                 #     tier="nova",
#                 #     model="2-general"
#                 # )
#             )
#         )

#     # Use Eleven Labs for Text-to-Speech
#     tts = ElevenLabsTTSService(
#       aiohttp_session=session,
#                 api_key=os.getenv("ELEVENLABS_API_KEY"),
#                 voice_id="zT03pEAEi0VHKciJODfn",
#                 model="eleven_multilingual_v1",
#       )

#     # Simple pipeline that will process text to speech and output the result
#     pipeline = Pipeline([tts, transport.output()])

#     # Create Pipecat processor that can run one or more pipelines tasks
#     runner = PipelineRunner()

#     # Assign the task callable to run the pipeline
#     task = PipelineTask(pipeline)

#     # Register an event handler to play audio when a
#     # participant joins the transport WebRTC session
#     @transport.event_handler("on_participant_joined")
#     async def on_new_participant_joined(transport, participant):
#       participant_name = participant["info"]["userName"] or ''
#       # Queue a TextFrame that will get spoken by the TTS service (Eleven Labs)
#       await task.queue_frames([TextFrame(f"Hello there, {participant_name}!"), EndFrame()])

#     # Run the pipeline task
#     await runner.run(task)

# if __name__ == "__main__":
#   asyncio.run(main())


#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import ast
import asyncio
import aiohttp
import os
import sys
import ast
import bot
from PIL import Image
# sys.path.append('/home/hrusheekesh.sawarkar/Reverie/sansadhak-platform/apps/twilio-chatbot/pipecat')
import pipecat
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from pipecat.frames.frames import (
    AudioRawFrame,
    ImageRawFrame,
    SpriteFrame,
    Frame,
    LLMMessagesFrame,
    TTSStoppedFrame
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.reverie_stt import ReverieSTTService

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from openai.types.chat import ChatCompletionToolParam
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.services.azure import AzureTTSService

from pipecat.transports.services.daily import DailyParams, DailyTranscriptionSettings, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from helpers import (
    TranscriptionLogger,
    TranscriptionRemoveSpace,
    LLMResposneLogger,
    ConversationEndManager,
    ReverieOpenAILLMService,
    KrutrimOpenAILLMService,
    TranslateInput,
)
from runner import configure
from pipecat.services.azure import AzureSTTService
from pipecat.services.reverie_tts import ReverieTTSService
from pipecat.services.dify import (
    DifyLLMService,
)
from loguru import logger
# import bot
from pipecat.processors.filters.function_filter import FunctionFilter
from dotenv import load_dotenv
load_dotenv(override=True)
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



# logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []

script_dir = os.path.dirname(__file__)

for i in range(1, 6):
    # Build the full path to the image file
    # full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    full_path = "pipecat/assests/robot01.png"
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(ImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

flipped = sprites[::-1]
sprites.extend(flipped)

# When the bot isn't talking, show a static image of the cat listening
quiet_frame = sprites[0]
talking_frame = SpriteFrame(images=sprites)


class TalkingAnimation(FrameProcessor):
    """
    This class starts a talking animation when it receives an first AudioFrame,
    and then returns to a "quiet" sprite when it sees a TTSStoppedFrame.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame)


async def main(room_url, token,bot_details):

    async with aiohttp.ClientSession() as session:
        # (room_url, token,bot_details) = await configure(session)
        bot_details=ast.literal_eval(bot_details)
        print(f'room_url:{room_url} token: {token}')
        print(f'bot_details:{bot_details}, bot_details type: {type(bot_details)}')
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                # camera_out_enabled=True,
                # camera_out_width=1024,
                # camera_out_height=576,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
                # transcription_settings=AzureSTTService(api_key=os.getenv("AZURE_API_KEY"), region=os.getenv("AZURE_REGION"), language="kn-IN")
                #
                # Spanish
                #
                # transcription_settings=DailyTranscriptionSettings(
                #     language="es",
                #     tier="nova",
                #     model="2-general"
                # )
            )
        )

        # log the bot details
        logger.info(f"Bot details in Run Bot: {bot_details}")

        # available_languages = bot_details.available_languages

        api_details = bot_details.get("api_details", [])
        logger.info(f"available language1 {api_details}")
        available_languages = api_details.get("available_languages", [])
        logger.info(f"available language1 {available_languages}")

        ivrDetails = bot_details.get("ivrDetails", {})
        botType=ivrDetails.get("botType")
        nmt_flag=ivrDetails.get("nmt")
        logger.info(f"IVR Details: {ivrDetails}")


        # logger.info(f"available language1 {api_details}")
        available_languages = api_details.get("available_languages", [])
        # logger.info(f"available language1 {available_languages}")

        selected_stt_providers = bot_details.get("providerData",{}).get("selected", {}).get("ivr", {}).get("stt", {})
        # logger.info(f"selected_stt_providers: {selected_stt_providers}")
        serializer_stt_provider = selected_stt_providers.get(available_languages[0],"")
        # logger.info(f"serializer_stt_provider: {serializer_stt_provider}")

        if serializer_stt_provider == "reverie":
            VAD_sample_Rate = 8000
        else:
            VAD_sample_Rate = 16000

        # ivrDetails = bot_details.get("ivrDetails", {})
        # logger.info(f"IVR Details: {ivrDetails}")

        providerData = bot_details.get("providerData", {})
        # logger.info(f"PROVIDER DATA: {providerData}")
        # botType = ivrDetails.get("botType")
        # nmt_flag = ivrDetails.get("nmt")
        nmt_provider = ivrDetails.get("nmtProvider")
        logger.debug(f"NMT: {nmt_flag}, NMT Provider: {nmt_provider }")
        user_details = bot_details.get("user_details", {})

        # callProvider= ivrDetails.get("callProvider")

        # logger.info(f"Call Provider: {callProvider}")
        logger.info(f"Bot Type: {botType}")

        agentSettings = bot_details.get("agentSettings", {})
        # logger.info(f"AGENT SETTINGS: {agentSettings}")
        callProvider = agentSettings.get("call", {}).get("callProvider", "")
        llmProvider = agentSettings.get("llm", {}).get("llmProvider", "")
        llmModel = "gpt-4o"  # Default model

        # conv_id = None #conv id is global used inside 2 api calls

        if llmProvider == "openai":
            llmModel = agentSettings.get("llm", {}).get("llmModel", llmModel)

        logger.info(f"Selected LLM Model: {llmProvider,llmModel}")

        # logger.info(f"Logger Info: {stt_pipeline}")

        global current_language
        current_language = "en"
        # current_language = "hi"
        # write a function which will change the global variable current_language, and this function will be called from the llm service
        def set_current_language(language):
            # log that the current language is being changed
            logger.debug(f"Changing the current language to: {language}")

            global current_language
            current_language = language
        
        

        # initialize messages
        messages = []

        def save_bot_context(messages):            
            bot_context_messages = messages

            # log the bot context messages
            logger.info(f"Bot Context Messages: {bot_context_messages}")

        if ivrDetails and ivrDetails.get("botType") == "dify-element":

                # get the dify api key
                difyApiKey = ivrDetails.get("difyApiKey")
                logger.info(f"Dify API Key: {difyApiKey}")

                bot_lang = available_languages[0]

                # set the current language
                # bot.set_current_language(bot_lang)
                bot_lang="hi"

                # get current language dify token
                language_dify_token = difyApiKey.get(bot_lang)
                logger.info(f"Language Dify Token: {language_dify_token}")

                # intelligent HR hindi assistant
                dify_llm = DifyLLMService(
                    aiohttp_session=session, api_key=language_dify_token,save_bot_context=save_bot_context,src_lan=bot_lang,nmt_flag=nmt_flag
                )           

                rev_pipecat_llm = dify_llm

                # based on the language set the message in messages
                if bot_lang == "hi":
                    messages = [{"role": "system", "content": "नमस्ते।"}]
                elif bot_lang == "en":
                    messages = [{"role": "system", "content": "Hello."}]
                elif bot_lang == "bn":
                    messages = [{"role": "system", "content": "নমস্কার।"}]
                elif bot_lang == "as":
                    messages = [{"role": "system", "content": "নমস্কাৰ।"}]
                elif bot_lang == "kn":
                    messages = [{"role": "system", "content": "ಹಲೋ।"}]
                elif bot_lang == "ml":
                    messages = [{"role": "system", "content": "ഹലോ।"}]
                elif bot_lang == "mr":
                    messages = [{"role": "system", "content": "नमस्कार।"}]
                elif bot_lang == "or":
                    messages = [{"role": "system", "content": "ନମସ୍କାର।"}]
                elif bot_lang == "ta":
                    messages = [{"role": "system", "content": "வணக்கம்।"}]
                elif bot_lang == "te":
                    messages = [{"role": "system", "content": "హలో।"}]
                elif bot_lang == "pa":
                    messages = [{"role": "system", "content": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ।"}]
                elif bot_lang == "gu":
                    messages = [{"role": "system", "content": "નમસ્તે।"}]

                tools = [
                    ChatCompletionToolParam(
                        type="function",
                        function={
                            "name": "conversation_end",
                            "description": "Funnction to end the call when conversation ends or user wants to end the call or user is busy",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "call_sid": {
                                        "type": "string",
                                        "description": "The call_sid that is being passed to the function.",
                                        "default": "call_sid",
                                    }
                                },
                                "required": ["call_sid"],
                            },
                        },
                    ),
                ]
                
                context = OpenAILLMContext(messages, tools)
                user_context = LLMUserContextAggregator(context)
                assistant_context = LLMAssistantContextAggregator(context)

        elif botType == "reverie-llm":
            bot_lang = available_languages[0]

            # set the current language
            set_current_language(bot_lang)

            agent_welcome_message = agentSettings.get("agent", {}).get("message")
            agentPrompt = agentSettings.get("agent", {}).get("prompt")
            agentPing = agentSettings.get("agent", {}).get("ping")
            
            logger.info(f"Agent Welcome Message: {agent_welcome_message}")
            
            # get name and constituency from user details
            name = user_details.get("name", "")
            constituency = user_details.get("constituency", "")
            
            # log the name and constituency
            logger.info(f"Name: {name}, Constituency: {constituency}")
            
            if name:
                agentPrompt = agentPrompt.replace("Rakesh", name)
            if constituency:
                agentPrompt = agentPrompt.replace("Karnal", constituency)

            # log the agent prompt
            # logger.info(f"Agent Prompt: {agentPrompt}")
            logger.debug(f"nmt_flag: {nmt_flag}, tgt_lan: {bot_lang}, nmt_provider: {nmt_provider}")
            # reverie openai llm service
            llm = ReverieOpenAILLMService(
                api_key=os.getenv("OPENAI_API_KEY"),
                # model="gpt-3.5-turbo",
                model="gpt-4o",
                # model=llmModel,
                save_bot_context=save_bot_context,
                # set_first_message_received=set_first_message_received,
                tgt_lan=bot_lang,
                nmt_flag=nmt_flag,
                nmt_provider=nmt_provider
            )
            

            
            # tools calling test
            messages = [
                {
                    "role": "system",
                    "content": agentPrompt,
                },
                {
                    "role":"assistant",
                    "content": agent_welcome_message
                },
                {
                    "role": "system",
                    "content": "Please introduce yourself and provide your name.",
                },
                # {
                #     "role": "system",
                #     "content": agentPing,
                # },
            ]

            tools = [
                ChatCompletionToolParam(
                    type="function",
                    function={
                        "name": "conversation_end",
                        "description": "Funnction to end the call when conversation ends or user wants to end the call or user is busy",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "call_sid": {
                                    "type": "string",
                                    "description": "The call_sid that is being passed to the function.",
                                    "default": "call_sid",
                                },
                                "stream_sid": {
                                    "type": "string",
                                    "description": "The stream_sid that is being passed to the function.",
                                    "default": "stream_sid",
                                },
                            },
                            "required": ["call_sid", "stream_sid"],
                        },
                    },
                ),
            ]

            # register conversation_end function
            # llm.register_function("conversation_end", conversation_end)

            rev_pipecat_llm = llm
            context = OpenAILLMContext(messages, tools)
            user_context = LLMUserContextAggregator(context)
            assistant_context = LLMAssistantContextAggregator(context)

        stt_services = ReverieSTTService(
            api_key="84148cc0e57e75c7d1b1331bb99a2e94aa588d48",
            src_lang=bot_lang,
            domain="generic-indocord",
        )

        # elevenlabs tts service for hindi language
        tts_11labs_hi = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="zT03pEAEi0VHKciJODfn",
            model="eleven_multilingual_v1",
        )

        tts_11labs_en = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            # voice_id="Xb7hH8MSUJpSbSDYk0k2",
            voice_id="zT03pEAEi0VHKciJODfn",
            model="eleven_turbo_v2",
        )




        async def english_language_filter(frame) -> bool:
            # log that the current language is being checked
            # logger.debug(f"Checking the current language: {current_language}")
            return current_language == "en"
        
    
        


        user_response = LLMUserResponseAggregator()
        assistant_response = LLMAssistantResponseAggregator()
        #call bot.py like this
        # stt = bot.create_stt_pipeline(avaial)
        ta = TalkingAnimation()

        pipeline = Pipeline([
            transport.input(),
            # azure_stt_kn,
            # deepgram_stt_hi,
            # deepgram_stt_en,
            stt_services,
            user_response,
            # llm,
            # krutim_llm,
            # dify_llm,
            rev_pipecat_llm,
            # tts,
            # reverie_tts_kn,
            tts_11labs_hi,
            # azure_tts_hi,
            # reverie_tts_hi,
            ta,
            transport.output(),
            assistant_response,
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
        await task.queue_frame(quiet_frame)
        # messages=[
        # #     {
        # #         "role": "system",
        # #         #
        # #         # English
        # #         #
        # #         "content": "You are Chatbot.",
        # #         # "content": """### Context
        # #         }
        # ]
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    # asyncio.run(main(room_url, token,bot_details))
    import sys
    if len(sys.argv) == 4:  # Expecting three arguments
        asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
    else:
        print("Please provide a room url, token, and bot details.")

