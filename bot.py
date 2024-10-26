import aiohttp
import os
import sys
import json
import uuid
import hashlib
import time

from pipecat.frames.frames import TextFrame, EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService, AzureOpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.reverie_stt import ReverieSTTService

from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.processors.frameworks.langchain import LangchainProcessor

from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.exotel import ExotelFrameSerializer
from openai.types.chat import ChatCompletionToolParam
from twilio.rest import Client
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from loguru import logger
# from loguru import logger as org_logger
from dotenv import load_dotenv
from helpers import (
    TranscriptionLogger,
    TranscriptionRemoveSpace,
    LLMResposneLogger,
    ConversationEndManager,
    ReverieOpenAILLMService,
    KrutrimOpenAILLMService,
    TranslateInput,
    InterruptionHandler,
    IgnorePacketsUntilFirstTTSPacketReceived,
    FirstTTSAudioEndProcessor,
    WelcomeMessageLLMResponseProcessor,
)
from pipecat.services.dify import (
    DifyLLMService,
)
from pipecat.services.rev_chatter import ReverieChatterLLMService
# from pipecat.services.playht import PlayHTTTSService
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.services.reverie_tts import ReverieTTSService
from pipecat.services.azure import AzureTTSService
from pipecat.services.azure import AzureSTTService
from pipecat.services.azure import AzureLLMService
from pipecat.services.google import GoogleSTTService , GoogleTTSService
from pipecat.services.knowledgebase import ReverieKnowledgeBase
from pipecat.transports.services.daily import DailyParams, DailyTransport
from datetime import datetime



try:
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        LiveOptions,
    )
except ModuleNotFoundError as e:
    # org_logger.error(f"Exception: {e}")
    # org_logger.error(    
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`. Also, set `DEEPGRAM_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class BotStartedSpeakingProcessor:
    def __init__(self):
        self.first_tts_audio_received = False

    # Getter for first_tts_audio_received
    def get_first_tts_audio_received(self):
        return self.first_tts_audio_received

    # Setter for first_tts_audio_received
    def set_first_tts_audio_received(self, value):
        self.first_tts_audio_received = value


load_dotenv(override=True)

# Definet the global websocket
STREAM_SID_WS = {}
STREAM_SID_CONVERSATION = {}


# Define the twilio call sid as global variable
global_twilio_call_sid = None
golbal_exotel_stream_id = None

# Initialize SileroVADAnalyzer globally
global_vad_analyzer = SileroVADAnalyzer()

# Initialize SileroVADAnalyzer for 8000 Hz sample rate globally
global_vad_8000_analyzer = SileroVADAnalyzer(sample_rate=8000)


# Add a new function for conversation end
async def conversation_end(llm, args):
    logger.debug(args)
    logger.debug(f"Conversation ends")
    # call_sid = args.get("call_sid")

    # provider = "exotel"
    provider = "twilio"

    if provider == "twilio":
        # take the call_sid from the global variable
        call_sid = global_twilio_call_sid

        if call_sid is None:
            raise Exception("call_sid is missing in args")

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

            # stop the current twilio call
            return {"message": "Goodbye! Thank you for using the chatbot."}

        except Exception as e:
            raise Exception(f"Failed to forward call: {str(e)}")

    elif provider == "exotel":
        if "stream_sid" in args:
            logger.debug("trying to end the call ...")
            if args["stream_sid"] not in STREAM_SID_WS:
                logger.debug(f"stream sid {args['stream_sid']} not found ...")
                return
            data_to_send = {"event": "stop", "stream_sid": args["stream_sid"]}
            logger.debug(f"sending payload {data_to_send}")
            await STREAM_SID_WS[args["stream_sid"]].send_json(json.dumps(data_to_send))
            logger.debug("call should end ...")


message_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]


async def run_bot(websocket_client, stream_sid, call_sid, bot_details, **kwargs):
    try:
        # log_handler_id = logger.add(
        #     f"call_logs/pipecat_{call_sid}.log", level="TRACE"
        # )  # Writing the log to a file

        name = None
        constituency = None
        if "custom_parameters" in kwargs:
            name = kwargs["custom_parameters"]["name"]
            constituency = kwargs["custom_parameters"]["constituency"]

        # call_provider = "exotel"
        # call_provider = callProvider or "twilio"
        call_provider = "twilio" # removing the callProvider since we know for sure if the call is coming from /media the call is coming from exotel
        # and we can take that info form the kwargs dict
        if "provider" in kwargs:
            call_provider = kwargs["provider"]


        # print the stream_sid as debug log
        logger.debug(f"stream_sid: {stream_sid}")
        # print the call_sid as debug log
        logger.debug(f"call_sid: {call_sid}")

        # log the bot details
        # logger.info(f"Bot details in Run Bot: {bot_details}")

        # available_languages = bot_details.available_languages

        api_details = bot_details.get("api_details", [])
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

        ivrDetails = bot_details.get("ivrDetails", {})
        # logger.info(f"IVR Details: {ivrDetails}")

        providerData = bot_details.get("providerData", {})
        # logger.info(f"PROVIDER DATA: {providerData}")
        botType = ivrDetails.get("botType")
        nmt_flag = ivrDetails.get("nmt")
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

        global global_twilio_call_sid
        global_twilio_call_sid = call_sid

        global STREAM_SID_WS
        STREAM_SID_WS[stream_sid] = websocket_client

        # global golbal_exotel_stream_id
        golbal_exotel_stream_id = stream_sid

        global current_language
        current_language = ""
        # current_language = "hi"

        # initialize messages
        messages = []

        async def english_language_filter(frame) -> bool:
            # log that the current language is being checked
            # logger.debug(f"Checking the current language: {current_language}")
            return current_language == "en"

        async def hindi_language_filter(frame) -> bool:
            # log that the current language is being checked
            # logger.debug(f"Checking the current language: {current_language}")
            return current_language == "hi"

        async def hindi_tts_language_filter(frame) -> bool:
            # log that the current language is being checked
            # logger.debug(f"Checking the current language: {current_language}")
            return current_language == "hi" or current_language == "choice"

        # function to set language choice filter
        async def choice_language_filter(frame) -> bool:
            return current_language == "choice"

        # bengali language filter
        async def bengali_language_filter(frame) -> bool:
            return current_language == "bn"

        # assamese language filter
        async def assamese_language_filter(frame) -> bool:
            return current_language == "as"

        # kannada language filter
        async def kannada_language_filter(frame) -> bool:
            return current_language == "kn"

        # malayalam language filter
        async def malayalam_language_filter(frame) -> bool:
            return current_language == "ml"

        # marathi language filter
        async def marathi_language_filter(frame) -> bool:
            return current_language == "mr"

        # odia language filter
        async def odia_language_filter(frame) -> bool:
            return current_language == "or"

        # tamil language filter
        async def tamil_language_filter(frame) -> bool:
            return current_language == "ta"

        # telugu language filter
        async def telugu_language_filter(frame) -> bool:
            return current_language == "te"

        # punjabi language filter
        async def punjabi_language_filter(frame) -> bool:
            return current_language == "pa"

        # gujarati language filter
        async def gujarati_language_filter(frame) -> bool:
            return current_language == "gu"

        # arabic language filter
        async def arabic_language_filter(frame) -> bool:
            return current_language == "ar"

        # write a function which will change the global variable current_language, and this function will be called from the llm service
        def set_current_language(language):
            # log that the current language is being changed
            logger.debug(f"Changing the current language to: {language}")

            global current_language
            current_language = language

        # add global variable to check if we have received first message from llm
        first_message_received = False

        # add parallel pipeline filter to check if we have received first message from llm
        async def first_message_filter(frame) -> bool:
            return first_message_received

        # function which will be called from llm service to set first message received
        def set_first_message_received():
            global first_message_received
            first_message_received = True
            logger.debug(f"First message received: {first_message_received}")

        # function to get the frame serializer
        def get_frame_serializer(call_provider, stream_sid):
            print(f"call provider: {call_provider}, stream_sid: {stream_sid}, ")

            if call_provider == "exotel":
                return ExotelFrameSerializer(stream_sid,serializer_stt_provider)
            elif call_provider == "plivo":
                return PlivoFrameSerializer(stream_sid,serializer_stt_provider)
            elif call_provider == "twilio":
                return TwilioFrameSerializer(stream_sid,serializer_stt_provider)
            else:
                return TwilioFrameSerializer(stream_sid,serializer_stt_provider)

        bot_context_messages = []

        # function to save the bot context
        def save_bot_context(messages):
            bot_context_messages = []

            for message in messages:
                timestamped_message = message.copy()
                timestamped_message['timestamp'] = datetime.utcnow().isoformat() + 'Z'  #UTC timestamp
                bot_context_messages.append(timestamped_message)

            # log the bot context messages
            # logger.info(f"Bot Context Messages: {bot_context_messages}")
            logger.debug(f"Bot Context Messages: ... {bot_context_messages[-1]}")
            STREAM_SID_CONVERSATION[stream_sid] = bot_context_messages

        async def save_conversation(bot_context_messages, bot_details, call_sid, provider):

            # Generate conv_id
            # conv_id = uuid.uuid4().hex[:16]
            conv_id = hashlib.md5(call_sid.encode()).hexdigest()[:16]

            # Extract template_id from bot_details
            template_id = bot_details["api_details"].get("TEMPLATE", "default_template")
            project_id = bot_details["api_details"].get("PROJECT", "default_project")

            url = "http://172.18.0.55:8001/save_dify_conversation"

            # Prepare the data payload
            data = {
                "conv_id": conv_id,
                "template_id": project_id,
                "conversation": bot_context_messages,
                "call_sid" : call_sid,
                "provider" : provider
            }

            # log the data
            logger.info(f"Data: {data}")

            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data),
                ) as response:
                    response_data = await response.text()
                    logger.info(f"API Response: {response_data}")

            return response_data

        async def call_summarize_api(bot_context_messages, call_sid, provider):
            # conv_id = uuid.uuid4().hex[:16]
            conv_id = hashlib.md5(call_sid.encode()).hexdigest()[:16]
            try:
                url = "http://172.18.0.55:8005/chat_summarizer"

                # Prepare the payload
                data = {
                    "conv_json": {"conversation": bot_context_messages},
                    "conv_id": conv_id,
                    "call_sid": call_sid,
                    "response_type": "both",
                    "provider": provider
                }
                
                logger.info(f"Summary api payload: {data}")

                # Make the API call
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(data),
                    ) as response:
                        response_data = await response.text()
                        logger.info(f"Summarize API Response: {response_data}")

                        return response_data

            except Exception as e:
                logger.error(f"Failed to call Summarize API: {str(e)}")
                return None

        agent_welcome_message = ""

        async with aiohttp.ClientSession() as session:
            vad_analyzer = global_vad_8000_analyzer if VAD_sample_Rate == 8000 else global_vad_analyzer

            transport = FastAPIWebsocketTransport(
                websocket=websocket_client,
                params=FastAPIWebsocketParams(
                    audio_out_enabled=True,
                    add_wav_header=False,
                    vad_enabled=True,
                    vad_analyzer=vad_analyzer,
                    vad_audio_passthrough=True,
                    serializer=(get_frame_serializer(call_provider, stream_sid)),
                ),
            )

            # transport_params = transport._params
            # serializer = transport_params.serializer

            rev_pipecat_llm = None
            context = None
            user_context = None
            assistant_context = None

            # llm = ReverieOpenAILLMService(
            #     api_key=os.getenv("OPENAI_API_KEY"),
            #     model="gpt-3.5-turbo",
            # )

            # krutim_llm = KrutrimOpenAILLMService(
            #     api_key="K5P_f+Dy9Z_lzs3.R_6XklzW",
            #     model="Krutrim-spectre-v2",
            #     base_url="https://cloud.olakrutrim.com/v1",
            # )

            # if ivrDetails and ivrDetails.get("botType") == "dify-element":

            if botType == "dify-element":

                # get the dify api key
                difyApiKey = ivrDetails.get("difyApiKey")
                logger.info(f"Dify API Key: {difyApiKey}")

                bot_lang = available_languages[0]

                # set the current language
                set_current_language(bot_lang)

                # get current language dify token
                language_dify_token = difyApiKey.get(bot_lang)
                logger.info(f"Language Dify Token: {language_dify_token}")

                # intelligent HR hindi assistant
                dify_llm = DifyLLMService(
                    aiohttp_session=session,
                    api_key=language_dify_token,
                    save_bot_context= save_bot_context,
                    tgt_lan=bot_lang,
                    nmt_flag=nmt_flag,
                    nmt_provider=nmt_provider
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
                                        "default": call_sid,
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

            elif botType == "rev-chatter":
                rev_chatter_llm = ReverieChatterLLMService(
                    bot_details=bot_details,
                    set_current_language=set_current_language,
                )

                rev_pipecat_llm = rev_chatter_llm

                user_context = LLMUserResponseAggregator(messages)
                assistant_context = LLMAssistantResponseAggregator(messages)

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
                
                # llm = BaseOpenAILLMServiceWithCache(
                #     api_key=os.getenv("OPENAI_API_KEY"),
                #     model="gpt-4o",
                #     save_bot_context=save_bot_context,
                #     # set_first_message_received=set_first_message_received,
                #     tgt_lan=bot_lang,
                #     nmt_flag=nmt_flag,
                #     nmt_provider=nmt_provider
                # )
                
                # llm = AzureOpenAILLMService(
                #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                #     endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                #     model="gpt-4o-mini",
                #     api_version="2024-02-15-preview",
                #     save_bot_context=save_bot_context,
                #     tgt_lan=bot_lang,
                #     nmt_flag=nmt_flag,
                #     nmt_provider=nmt_provider
                # )
                
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
                    # {
                    #     "role": "system",
                    #     "content": "Please introduce yourself and provide your name.",
                    # },
                    {
                        "role": "system",
                        "content": agentPing,
                    },
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
                                        "default": call_sid,
                                    },
                                    "stream_sid": {
                                        "type": "string",
                                        "description": "The stream_sid that is being passed to the function.",
                                        "default": stream_sid,
                                    },
                                },
                                "required": ["call_sid", "stream_sid"],
                            },
                        },
                    ),
                ]

                # register conversation_end function
                llm.register_function("conversation_end", conversation_end)

                rev_pipecat_llm = llm
                context = OpenAILLMContext(messages, tools)
                user_context = LLMUserContextAggregator(context)
                assistant_context = LLMAssistantContextAggregator(context)
            
            elif botType == "reverie-azure-llm":
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

                # reverie openai llm service
                # llm = ReverieOpenAILLMService(
                #     api_key=os.getenv("OPENAI_API_KEY"),
                #     # model="gpt-3.5-turbo",
                #     model="gpt-4o",
                #     # model=llmModel,
                #     save_bot_context=save_bot_context,
                #     # set_first_message_received=set_first_message_received,
                #     tgt_lan=bot_lang,
                #     nmt_flag=nmt_flag,
                #     nmt_provider=nmt_provider
                # )
                
                # llm = BaseOpenAILLMServiceWithCache(
                #     api_key=os.getenv("OPENAI_API_KEY"),
                #     model="gpt-4o",
                #     save_bot_context=save_bot_context,
                #     # set_first_message_received=set_first_message_received,
                #     tgt_lan=bot_lang,
                #     nmt_flag=nmt_flag,
                #     nmt_provider=nmt_provider
                # )
                
                llm = AzureOpenAILLMService(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    model="gpt-4o-mini",
                    api_version="2024-02-15-preview",
                    save_bot_context=save_bot_context,
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
                    # {
                    #     "role": "system",
                    #     "content": "Please introduce yourself and provide your name.",
                    # },
                    {
                        "role": "system",
                        "content": agentPing,
                    },
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
                                        "default": call_sid,
                                    },
                                    "stream_sid": {
                                        "type": "string",
                                        "description": "The stream_sid that is being passed to the function.",
                                        "default": stream_sid,
                                    },
                                },
                                "required": ["call_sid", "stream_sid"],
                            },
                        },
                    ),
                ]

                # register conversation_end function
                llm.register_function("conversation_end", conversation_end)

                rev_pipecat_llm = llm
                context = OpenAILLMContext(messages, tools)
                user_context = LLMUserContextAggregator(context)
                assistant_context = LLMAssistantContextAggregator(context)
            
            elif botType == "knowledge-base":

                agent_welcome_message = agentSettings.get("agent", {}).get("message")
                agentPrompt = agentSettings.get("agent", {}).get("prompt")
                
                logger.info(f"Agent Welcome Message: {agent_welcome_message}")

                logger.debug(f"available_languages[0]: {available_languages[0]}")
                bot_lang = available_languages[0]
                # set the current language
                set_current_language(bot_lang)

                collection_name = '53f0f281-ab6a-4af1-96bd-a39734964960'

                logger.debug(f"in reverie-kb")
                llm = ReverieKnowledgeBase(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    # model="gpt-3.5-turbo",
                    model="gpt-4o",
                    namespace=collection_name,
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
                                        "default": call_sid,
                                    },
                                    "stream_sid": {
                                        "type": "string",
                                        "description": "The stream_sid that is being passed to the function.",
                                        "default": stream_sid,
                                    },
                                },
                                "required": ["call_sid", "stream_sid"],
                            },
                        },
                    ),
                ]

                # register conversation_end function
                llm.register_function("conversation_end", conversation_end)
                rev_pipecat_llm = llm
                context = OpenAILLMContext(messages, tools)
                user_context = LLMUserContextAggregator(context)
                assistant_context = LLMAssistantContextAggregator(context)


            else:
                rev_chatter_llm = ReverieChatterLLMService(
                    bot_details=bot_details,
                    set_current_language=set_current_language,
                )

                rev_pipecat_llm = rev_chatter_llm
                user_context = LLMUserResponseAggregator(messages)
                assistant_context = LLMAssistantResponseAggregator(messages)

            # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

            # stt for english language
            # deepgram_stt_en = DeepgramSTTService(
            #     api_key=os.getenv("DEEPGRAM_API_KEY"),
            #     url="",
            #     live_options=LiveOptions(
            #         encoding="linear16",
            #         language="en-US",
            #         model="nova-2-conversationalai",
            #         sample_rate=16000,
            #         channels=1,
            #         interim_results=True,
            #         smart_format=True,
            #         # numerals=True,
            #     ),
            #     language="en",
            # )

            # deepgram hindi stt service
            # deepgram_stt_hi = DeepgramSTTService(
            #     api_key=os.getenv("DEEPGRAM_API_KEY"),
            #     url="",
            #     live_options=LiveOptions(
            #         encoding="linear16",
            #         language="hi",
            #         model="enhanced-general",
            #         sample_rate=16000,
            #         channels=1,
            #         interim_results=True,
            #         numerals=True,
            #     ),
            #     language="hi",
            # )

            # reverie stt for all languages mentioned in allLanguages
            # allLanguages = {
            #     "en": "English",
            #     "hi": "Hindi",
            #     "bn": "Bengali",
            #     "as": "Assamese",
            #     "kn": "Kannada",
            #     "ml": "Malayalam",
            #     "mr": "Marathi",
            #     "or": "Odia",
            #     "ta": "Tamil",
            #     "te": "Telugu",
            #     "pa": "Punjabi",
            #     "ar": "Arabic",
            #     "gu": "Gujarati",
            # }

            # playht tts service
            # tts_playht = PlayHTTTSService(
            #     api_key=os.getenv("PLAY_HT_API_KEY"),
            #     user_id=os.getenv("PLAY_HT_USER_ID"),
            #     voice_url="s3://voice-cloning-zero-shot/f3c22a65-87e8-441f-aea5-10a1c201e522/original/manifest.json",
            # )

            # Google STT services for all languages
            # google_stt_services = {}
            # allLanguages = [
            #     "hi",
            #     "en",
            #     "bn",
            #     "as",
            #     "kn",
            #     "ml",
            #     "mr",
            #     "or",
            #     "ta",
            #     "te",
            #     "pa",
            #     "gu",
            # ]

            # for lang in allLanguages:
            #     google_stt_services[lang] = GoogleSTTService(
            #         api_key=os.getenv("GOOGLE_API_KEY"),
            #         language_code=f"{lang}-IN" if lang != "en" else "en-US",
            #     )
            #     await google_stt_services[lang].initialize()

            # logger.info(
            #     f"Initialized Google STT services for languages: {', '.join(allLanguages)}"
            # )

            # Google TTS services for all languages
            # google_tts_services = {}

            # for lang in allLanguages:
            #     language_code = f"{lang}-IN" if lang != "en" else "en-US"
            #     if lang == "kn":
            #         # voice_name = "kn-IN-Wavenet-D"
            #         voice_name = "kn-IN-Wavenet-C"
            #     else:
            #         voice_name = f"{language_code}-Standard-A"  # Using a standard voice for each language

            #     google_tts_services[lang] = await GoogleTTSService.create(
            #         voice_name=voice_name, language_code=language_code
            #     )

            # logger.info(
            #     f"Initialized Google TTS services for languages: {', '.join(allLanguages)}"
            # )

            # playht tts service for all languages
            # playht_tts_services = {}
            # for lang in allLanguages:
            #     playht_tts_services[lang] = PlayHTTTSService(
            #         api_key=os.getenv("PLAY_HT_API_KEY"),
            #         user_id=os.getenv("PLAY_HT_USER_ID"),
            #         voice_url="s3://voice-cloning-zero-shot/f3c22a65-87e8-441f-aea5-10a1c201e522/original/manifest.json",
            #     )
            # logger.info(
            #     f"Initialized PlayHT TTS services for languages: {', '.join(allLanguages)}"
            # )

            # get the transscription logger
            # transcription_logger = TranscriptionLogger()

            # transacription remove space
            # transcription_remove_space = TranscriptionRemoveSpace()

            # Get the llm response logger
            # llm_response_logger = LLMResposneLogger()

            # translate user response
            # translate_user_response = TranslateInput(src_lang="hi", tgt_lang="en")

            # translate llm response
            # translate_llm_response = TranslateInput(src_lang="en", tgt_lang="hi")

            async def save_conversation_callback():
                try:
                    if stream_sid in STREAM_SID_CONVERSATION:
                        bot_conversation = STREAM_SID_CONVERSATION[stream_sid]
                        logger.debug("------ trying to save conversation --------")
                        logger.info(f"Bot Context Messages: {bot_conversation}")

                        await save_conversation(bot_conversation, bot_details, call_sid, call_provider)
                    else:
                        logger.debug(f"------ conversation not found {stream_sid} --------")
                except Exception as e:
                    logger.error(f"Error in save_conversation_callback: {e}")

            async def summarise_conversation_callback():
                try:
                    if stream_sid in STREAM_SID_CONVERSATION:
                        bot_conversation = STREAM_SID_CONVERSATION[stream_sid]
                        logger.debug("------ trying to save summary --------")
                        logger.info(f"Bot Context Messages: {bot_conversation}")

                        await call_summarize_api(bot_conversation, call_sid, call_provider)
                    else:
                        logger.debug(f"------ conversation not found {stream_sid} --------")

                    logger.info(f"Bot Context Messages: {bot_conversation}")
                except Exception as e:
                    logger.error(f"Error in summarise_conversation_callback: {e}")

            # Get the conversation end manager
            conv_end_mngr_pipeline_start_time = time.time()
            conversation_end_manager = ConversationEndManager(
                websocket_client=websocket_client,
                call_sid=call_sid,
                stream_sid=stream_sid,
                save_conversation_callback=save_conversation_callback,
                summarise_conversation_callback=summarise_conversation_callback,
                call_provider=call_provider,
            )
            logger.info("conv_end_mngr pipeline Time consuming: {:.4f}s".format(time.time() - conv_end_mngr_pipeline_start_time))
            
            # welcome message manager
            wlcm_msg_pipeline_start_time = time.time()
            welcome_message_manager = WelcomeMessageLLMResponseProcessor(welcome_message = agent_welcome_message)
            logger.info("wlcm_msg pipeline Time consuming: {:.4f}s".format(time.time() - wlcm_msg_pipeline_start_time))

            # user_context = LLMUserResponseAggregator(messages)
            # assistant_context = LLMAssistantResponseAggregator(messages)

            # avt = AudioVolumeTimer()
            # tl = TranscriptionTimingLogger(avt)

            # context = OpenAILLMContext(messages, tools)
            # user_context = LLMUserContextAggregator(context)
            # assistant_context = LLMAssistantContextAggregator(context)

            int_hndler_pipeline_start_time = time.time()
            interruption_handler = InterruptionHandler(
                websocket_client, stream_sid, call_sid, call_provider
            )
            logger.info("Interruption handler pipeline Time consuming: {:.4f}s".format(time.time() - int_hndler_pipeline_start_time))

            # ignore_packet_handler = IgnorePacketsUntilLLMResponse()
            # ignore_packet_handler = IgnoreAudioUntilTTS()

            # ----------------- LANGCHAIN -----------------
            # async def user_idle_callback(user_idle: UserIdleProcessor):
            #     messages.append(
            #         {
            #             "role": "system",
            #             "content": "Ask the user if they are still there and try to prompt for some input, but be short.",
            #         }
            #     )
            #     await user_idle.queue_frame(LLMMessagesFrame(messages))
            # user_idle = UserIdleProcessor(callback=user_idle_callback, timeout=5.0)
            # ----------------- LANGCHAIN -----------------

            # ----------------- LANGCHAIN -----------------
            # prompt = ChatPromptTemplate.from_messages(
            #     [
            #         (
            #             "system",
            #             "Be nice and helpful. Answer very briefly and without special characters like `#` or `*`. "
            #             "Your response will be synthesized to voice and those characters will create unnatural sounds.",
            #         ),
            #         MessagesPlaceholder("chat_history"),
            #         ("human", "{input}"),
            #     ]
            # )
            # chain = prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            # history_chain = RunnableWithMessageHistory(
            #     chain,
            #     get_session_history,
            #     history_messages_key="chat_history",
            #     input_messages_key="input",
            # )
            # langchain = LangchainProcessor(history_chain)
            # ----------------- LANGCHAIN -----------------

            def create_stt_pipeline(available_languages):
                    language_filters = {
                        "en": english_language_filter,
                        "hi": hindi_language_filter,
                        "bn": bengali_language_filter,
                        "as": assamese_language_filter,
                        "kn": kannada_language_filter,
                        "ml": malayalam_language_filter,
                        "mr": marathi_language_filter,
                        "or": odia_language_filter,
                        "ta": tamil_language_filter,
                        "te": telugu_language_filter,
                        "pa": punjabi_language_filter,
                        "gu": gujarati_language_filter,
                        "ar": arabic_language_filter,
                    }

                    # Default STT services
                    stt_services = {}

                    # Fetch the STT provider data and selected providers from providerData
                    stt_provider_data = providerData.get("config", {}).get("ivr", {}).get("stt", {})
                    selected_stt_providers = providerData.get("selected", {}).get("ivr", {}).get("stt", {})

                    # Log the STT provider data and selected providers
                    # logger.info(f"stt_provider_data: {stt_provider_data}")
                    # logger.info(f"selected_stt_providers: {selected_stt_providers}")

                    # Update STT services based on selected providers
                    for lang, provider_name in selected_stt_providers.items():
                        logger.debug(f"stt provider : {provider_name}, lang {lang}, stt services {stt_services}")

                        provider_config = stt_provider_data.get(provider_name, {}).get(lang, {})

                        if provider_name == "deepgram":
                            logger.info("inside deepgram")
                            stt_services[lang] = DeepgramSTTService(
                                api_key=os.getenv("DEEPGRAM_API_KEY"),
                                url="",
                                live_options=LiveOptions(
                                    encoding="linear16",
                                    language=provider_config.get("language", f"{lang}-IN"),
                                    model=provider_config.get("model", "nova-2-conversationalai"),
                                    sample_rate=16000,
                                    channels=1,
                                    interim_results=True,
                                    smart_format=True,
                                ),
                                language=lang,
                            )
                        elif provider_name == "reverie":
                            logger.info("inside reverie")
                            stt_services[lang] = ReverieSTTService(
                                api_key="84148cc0e57e75c7d1b1331bb99a2e94aa588d48",
                                src_lang=lang,
                                domain=provider_config.get("domain", "generic"),
                            )
                        elif provider_name == "azure":
                            logger.info("inside azure")
                            
                            logger.debug(f"provider_config: {provider_config}")
                            
                            stt_services[lang] = AzureSTTService(
                                api_key=os.getenv("AZURE_API_KEY"),
                                region=os.getenv("AZURE_REGION"),
                                language=provider_config.get("language", f"{lang}-IN"),
                            )
                        elif provider_name == "google":
                            logger.info("inside google")
                            stt_services[lang] = GoogleSTTService(
                                api_key=os.getenv("GOOGLE_API_KEY"),
                                language_code=f"{lang}-IN" if lang != "en" else "en-US",
                            )

                    # Log the updated STT services
                    logger.info(f"Updated STT services: {stt_services}")

                    pipelines = []
                    logger.info(f"Available languages for STT: {available_languages}")
                    
                    # If only one language is available, return the STT service for that language
                    if len(available_languages) == 1:
                        lang = available_languages[0]
                        return stt_services[lang]

                    # If multiple languages are available, create a pipeline for each language
                    for lang in available_languages:
                        logger.info(f"Creating STT pipeline for language: {lang}")
                        if lang in language_filters and lang in stt_services:
                            pipelines.append(
                                [FunctionFilter(language_filters[lang]), stt_services[lang]]
                            )

                    # If multiple languages are available, add a language choice filter
                    if len(available_languages) > 1:
                        # initiate azure stt for lang choice
                        azure_stt_lang_choice = AzureSTTService(
                            api_key=os.getenv("AZURE_API_KEY"),
                            region=os.getenv("AZURE_REGION"),
                            language="en-US",
                            available_languages=available_languages,
                        )
                        
                        pipelines.append(
                            [FunctionFilter(choice_language_filter), azure_stt_lang_choice]
                        )

                    logger.info(f"STT pipeline created: {pipelines}")
                    return ParallelPipeline(*pipelines)

            stt_pipeline_start_time = time.time()
            stt_pipeline = create_stt_pipeline(available_languages)
            logger.info("STT pipeline Time consuming: {:.4f}s".format(time.time() - stt_pipeline_start_time))

            async def create_tts_pipeline(available_languages):
                language_filters = {
                    "en": english_language_filter,
                    "hi": hindi_tts_language_filter,
                    "bn": bengali_language_filter,
                    "as": assamese_language_filter,
                    "kn": kannada_language_filter,
                    "ml": malayalam_language_filter,
                    "mr": marathi_language_filter,
                    "or": odia_language_filter,
                    "ta": tamil_language_filter,
                    "te": telugu_language_filter,
                    "pa": punjabi_language_filter,
                    "gu": gujarati_language_filter,
                }

                tts_services = {}

                tts_provider_data = (
                    providerData.get("config", {}).get("ivr", {}).get("tts", {})
                )
                selected_tts_providers = (
                    providerData.get("selected", {}).get("ivr", {}).get("tts", {})
                )

                # log tts_provider_data and selected_tts_providers
                logger.info(f"tts_provider_data: {tts_provider_data}")
                logger.info(f"selected_tts_providers: {selected_tts_providers}")

                for lang, provider_name in selected_tts_providers.items():
                    provider_config = tts_provider_data.get(provider_name, {}).get(
                        lang, {}
                    )

                    if provider_name == "elevenlabs":
                        logger.info("inside eleven labs tts")
                        tts_services[lang] = ElevenLabsTTSService(
                            aiohttp_session=session,
                            api_key=os.getenv("ELEVENLABS_API_KEY"),
                            voice_id=provider_config.get(
                                "voice_id", "JNaMjd7t4u3EhgkVknn3"
                            ),
                            model=provider_config.get(
                                "model_id", "eleven_multilingual_v1"
                            ),
                        )
                    elif provider_name == "reverie":
                        logger.info("inside reverie tts")
                        tts_services[lang] = ReverieTTSService(
                            aiohttp_session=session,
                            api_key=os.getenv("REVERIE_API_KEY"),
                            # speaker=f"{lang}_female",  # or another dynamic value if provided
                            speaker= provider_config.get("speaker","hi_female"),
                            format="wav",
                            speed=provider_config.get("speed", 1.2),
                            pitch=provider_config.get("pitch", 1),
                        )
                    elif provider_name == "google":
                        logger.info("inside google tts")
                        language_code = f"{lang}-IN" if lang != "en" else "en-US"
                        voice_name = provider_config.get(
                            "voice", f"{language_code}-Standard-A"
                        )
                        tts_services[lang] = await GoogleTTSService.create(
                            voice_name=voice_name, language_code=language_code
                        )
                    elif provider_name == "azure":
                        logger.info("inside azure tts")
                        tts_services[lang] = AzureTTSService(
                            api_key=os.getenv("AZURE_API_KEY"),
                            region=os.getenv("AZURE_REGION"),
                            voice=provider_config.get("voice", "en-US-AriaRUS"),
                        )
                    # elif provider_name == "playht":
                    #     logger.info("inside playht tts")
                    #     voice_url = provider_config.get(
                    #         "voiceUrl",
                    #         "s3://voice-cloning-zero-shot/f3c22a65-87e8-441f-aea5-10a1c201e522/original/manifest.json"
                    #     )
                    #     tts_services[lang] = PlayHTTTSService(
                    #         api_key=os.getenv("PLAY_HT_API_KEY"),
                    #         user_id=os.getenv("PLAY_HT_USER_ID"),
                    #         voice_url=voice_url,
                    #     )
                # log tts services
                logger.info(f"tts_services: {tts_services}")

                pipelines = []
                logger.info(f"Available languages for TTS: {available_languages}")
                
                
                # If only one language is available, return the TTS service for that language
                if len(available_languages) == 1:
                    lang = available_languages[0]
                    return tts_services[lang]
                

                # If multiple languages are available, create a pipeline for each language
                for lang in available_languages:
                    logger.info(f"Creating TTS pipeline for language: {lang}")
                    if lang in language_filters and lang in tts_services:
                        pipelines.append(
                            [FunctionFilter(language_filters[lang]), tts_services[lang]]
                        )

                logger.info(f"TTS pipeline created: {pipelines}")
                return ParallelPipeline(*pipelines)


            tts_pipeline_start_time = time.time()
            tts_pipeline = await create_tts_pipeline(available_languages)
            logger.info("TTS pipeline Time consuming: {:.4f}s".format(time.time() - tts_pipeline_start_time))

            logger.info(f"stt_pipeline logs: {stt_pipeline}")
            logger.info(f"tts_pipeline logs: {tts_pipeline}")
            
            # async def user_idle_callback(user_idle: UserIdleProcessor):
            #     messages.append(
            #         {"role": "system", "content": "Ask the user next question that you have to continue the conversation"})
            #     await user_idle.queue_frame(LLMMessagesFrame(messages))
            # user_idle = UserIdleProcessor(callback=user_idle_callback, timeout=10.0)
                    
            bot_started_speaking = BotStartedSpeakingProcessor()

            # handle first tts audio end event
            async def handle_first_tts_audio_end(firstTtsAudioEnd: FirstTTSAudioEndProcessor):
                
                logger.info("Control has reached handle_first_tts_audio_end")
                
                # logger.info("------ tts audio end event --------")
                bot_started_speaking.set_first_tts_audio_received(True)
            
            firstTtsAudioEnd = FirstTTSAudioEndProcessor(callback=handle_first_tts_audio_end)
            
            ignore_packet_handler = IgnorePacketsUntilFirstTTSPacketReceived(audio_receive_callback = bot_started_speaking.get_first_tts_audio_received)

            pipeline = Pipeline(
                [
                    transport.input(),  # Websocket input from client
                    ignore_packet_handler, # Ignore packets until first tts audio received
                    stt_pipeline,
                    interruption_handler,  # Interruption handler
                    user_context,  # User responses
                    rev_pipecat_llm,  # LLM
                    welcome_message_manager,  # Welcome message manager
                    conversation_end_manager,  # this is required only when we are using dify llm to end the conversation
                    tts_pipeline,
                    firstTtsAudioEnd,
                    transport.output(),  # Websocket output to client
                    assistant_context,  # LLM responses
                ]
            )

            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                    # enable_metrics=True,
                    # report_only_initial_ttfb=True,
                ),
            )

            # log the context
            logger.info(f"Context: {context}")

            @transport.event_handler("on_client_connected")
            async def on_client_connected(transport, client):
                # Kick off the conversation.
                # messages.append(
                #     {
                #         "role": "system",
                #         "content": "Please introduce yourself to the user.",
                #     }
                # )
                await task.queue_frames([LLMMessagesFrame(messages)])

            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(transport, client):
                logger.debug("------ call disconnected --------")
                await task.queue_frames([EndFrame()])
                await save_conversation_callback()
                await summarise_conversation_callback()

            runner = PipelineRunner(handle_sigint=False)

            try:
                await runner.run(task)
            except Exception as e:
                logger.error(f"Error from runner: {str(e)}")

    except Exception as e:
        logger.error(f"Error from main exception: {str(e)}")
