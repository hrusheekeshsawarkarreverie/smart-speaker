import json
import os
import uvicorn
import urllib.parse
import requests
import subprocess
import argparse
from fastapi.responses import Response
from loguru import logger
from fastapi import FastAPI, WebSocket, Request, HTTPException, Query,Form,BackgroundTasks
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Gather  # type: ignore
from twilio.rest import Client
from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomParams,
)
from contextlib import asynccontextmanager
import plivo
from plivo import plivoxml

from bot import run_bot
import aiohttp
import json
import shutil
import uuid
import sys
import requests
import time
import redis

########################################
logger.remove(0)
logger.add(
    "logs/pipecat_server_standalone.log", level="DEBUG", rotation="6h"
)  # Writing the log to a file
logger.add(sys.stderr, level="DEBUG")
########################################


MAX_BOTS_PER_ROOM = 1

# Bot sub-process dict for status reporting and concurrency control
bot_procs = {}
daily_helpers = {}


def cleanup():
    # Clean up function, just to be extra safe
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


@asynccontextmanager
async def lifespan(app: FastAPI):
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        # aiohttp_session=aiohttp_session
    )
    print(
        os.getenv("DAILY_API_KEY", ""),
        os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
    )
    yield
    await aiohttp_session.close()
    cleanup()


parser = argparse.ArgumentParser(description="A script that runs different Setups based on arguments (WebRTC and IVR.")

# Adding an optional argument
parser.add_argument('--arg', type=str, help='An optional argument to trigger a specific function')

# Parsing the arguments
args = parser.parse_args()

print(args)
if args.arg:
    print("Starting WebRTC Setup")
    app = FastAPI(lifespan=lifespan)
else:
    print("Starting IVR Setup")
    app = FastAPI()

# app = FastAPI()


# use this for the IVR version

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CALL_DETAILS = dict()

# Fetch Twilio API Details
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)


# fetch Plivo API details
plivo_auth_id = os.getenv("PLIVO_AUTH_ID")
plivo_auth_token = os.getenv("PLIVO_AUTH_TOKEN")
plivo_phone_number = os.getenv("PLIVO_PHONE_NUMBER")
print(plivo_auth_id, plivo_auth_token)
# Initialize Plivo client
plivo_client = plivo.RestClient(
    os.getenv("PLIVO_AUTH_ID"), os.getenv("PLIVO_AUTH_TOKEN")
)


def format_stt_variables(variables):
    formatted_variables = dict()
    for variable in variables:
        formatted_variables[variable["name"]] = variable["sttConfig"]["domain"]

    # logger.debug(f"formatted variables : {formatted_variables}")
    return formatted_variables


async def get_bot_details_by_conversation_id(conversation_id):
    try:

        # if the conversation id is uuid type then it is not a valid conversation id
        if len(str(conversation_id)) == 16:
            # log that conversation id is not valid
            logger.warning(f"Conversation id is not valid: {conversation_id}")
            return None

        # log that control is comng in the else block
        logger.info(f"Getting bot details for conversation id: {conversation_id}")

        # check if the bot details are already stored in memory
        bot_details = get_bot_details_from_memory(conversation_id)
        if bot_details:
            # logger.debug(f"if Bot Detail: {bot_details}")
            return bot_details

        url = "https://sansadhak-dev.reverieinc.com/api/bot/deploy/details"
        payload = json.dumps({"conversationId": int(conversation_id)})
        headers = {
            "Content-Type": "application/json",
            "Origin": "https://sansadhak-dev.reverieinc.com",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                resp_obj = await response.json()
                logger.info(f"Got context for {conversation_id} => {resp_obj}")

                if "data" not in resp_obj:
                    return {
                    "api_details": {
                        "REV-APP-ID": "com.domain",
                        "REV-APPNAME": "nlu",
                        "REV-API-KEY": "732407ffce16f9362f9f0eeb2b5aa5758cd09039",
                        "PROJECT": "Eastman Auto",
                        "MODEL": "eastman_model",
                        "SUPPORT_PROJECT": "Eastman Auto",
                        "SUPPORT_MODEL": "eastman_model",
                        "TEMPLATE": "Eastman Auto_1720609506.0128822",
                        "available_languages": ["en", "hi"],
                    },
                    "stt_variables": {},
                    "tts_variables": {},
                    "selectLanguage": False,
                    }

                languages = resp_obj["data"]["testDetails"].get("languages", [])
                PROJECT = resp_obj["data"]["testDetails"].get("projectName", "")
                MODEL = resp_obj["data"]["testDetails"].get("modelName", "")
                TEMPLATE = resp_obj["data"]["testDetails"].get("templateName", "")

                stt_variables = format_stt_variables(
                    resp_obj["data"].get("sttVariablesInfo", [])
                )

                tts_variables = {}
                if "ttsProvider" in resp_obj["data"]["testDetails"]:
                    pass

                if "ttsSettings" in resp_obj["data"]["testDetails"]:
                    pass
                
                # log some message here
                logger.info(f"Languages: {languages}")

                try:
                    selectLanguage = (
                        resp_obj.get("data", {})
                        .get("botStyle", {})
                        .get("style", {})
                        .get("selectLanguage", True)
                    )
                except Exception as e:
                    logger.error(f"An error occurred while fetching selectLanguage: {str(e)}")
                    selectLanguage = True  # Default value if there's an error
                
                # log some message here
                logger.info(f"Select Language: {selectLanguage}")

                ivrDetails = resp_obj.get("data", {}).get("ivrDetails", {})
                providerData = resp_obj.get("data", {}).get("providerData", {})

                # log the ivr details
                logger.info(f"IVR details: {ivrDetails}")

                logger.info(f"PROVIDER DATA: {providerData}")

                agentSettings = resp_obj.get("data", {}).get("agentSettings", {})
                logger.info(f"Agent settings: {agentSettings}")

                response = {
                    "api_details": {
                    "REV-APP-ID": "com.domain",
                    "REV-APPNAME": "nlu",
                    "REV-API-KEY": "732407ffce16f9362f9f0eeb2b5aa5758cd09039",
                    "PROJECT": PROJECT,
                    "MODEL": MODEL,
                    "SUPPORT_PROJECT": PROJECT,
                    "SUPPORT_MODEL": MODEL,
                    "TEMPLATE": TEMPLATE,
                    "available_languages": languages,
                    },
                    "stt_variables": stt_variables,
                    "tts_variables": tts_variables,
                    "selectLanguage": selectLanguage,
                    "ivrDetails": ivrDetails,
                    "providerData": providerData,
                    "agentSettings": agentSettings,
                }

                # store the bot details in memory
                store_bot_details(conversation_id, response)

                return response
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # Handle the error here
        return None


# write a sample code to store json data in memory based on the conversation id
def store_bot_details(conversation_id, bot_details):
    try:
        # store the bot details in Redis with conversation_id as the key
        redis_client.set(conversation_id, json.dumps(bot_details))
    except Exception as e:
        logger.error(f"An error occurred while storing bot details in Redis: {str(e)}")


# function to get bot details from memory based on the conversation id
# Initialize Redis client
redis_host = os.getenv('REDIS_HOST', 'redis')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

def get_bot_details_from_memory(conversation_id):
    try:
        # log that we are getting bot details from memory
        logger.info(
            f"Getting bot details from memory for conversation id: {conversation_id}"
        )

        # get the bot details from Redis cache
        bot_details = redis_client.get(conversation_id)
        
        # log the bot_details
        logger.debug(f"Bot Details: {bot_details}")
        
        if bot_details:
            return json.loads(bot_details)

        logger.warning(
            f"Bot details not found in memory for conversation id: {conversation_id}"
        )
        return None
    except Exception as e:
        logger.error(
            f"An error occurred while getting bot details from memory: {str(e)}"
        )
        return None

# function to store user details based on unique id
def store_user_details(user_id, user_details):
    try:
        # store the user details in Redis with user_id as the key
        redis_client.set(user_id, json.dumps(user_details))
    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        # Retry logic or alternative handling can be added here
    except Exception as e:
        logger.error(f"An error occurred while storing user details in Redis: {str(e)}")


# function to get user details from memory based on the unique id
def get_user_details_from_memory(user_id):
    try:
        # log that we are getting user details from memory
        logger.info(f"Getting user details from memory for user id: {user_id}")

        # get the user details from Redis cache
        user_details = redis_client.get(user_id)
        if user_details:
            return json.loads(user_details)

        logger.warning(f"User details not found in memory for user id: {user_id}")
        return None
    except Exception as e:
        logger.error(
            f"An error occurred while getting user details from memory: {str(e)}"
        )
        return None


@app.post("/start_call")
async def start_call(request: Request):
    print("POST TwiML")
    stream_url = os.getenv("APPLICATION_BASE_URL")
    xml_content = open("templates/streams.xml").read()
    xml_content_with_url = xml_content.replace("{{ base_url }}", stream_url)

    return HTMLResponse(content=xml_content_with_url, media_type="application/xml")


@app.post("/start_call_with_pin")
async def start_call(request: Request):
    response = VoiceResponse()
    gather = Gather(num_digits=7, action="/enter_pin")
    gather.say("Please enter your 6-digit PIN, followed by the pound key.")
    response.append(gather)
    return PlainTextResponse(str(response), media_type="application/xml")


@app.post("/enter_pin")
async def enter_pin(request: Request):
    try:
        request_body = await request.body()
        data_str = request_body.decode("utf-8")
        parsed_data = urllib.parse.parse_qs(data_str)
        pin = parsed_data.get("Digits", [""])[0]
        pin = pin[:6]

        if pin:
            response = VoiceResponse()
            # response.say("PIN validated. Establishing connection.")
            response.redirect("/connect_call_with_pin?pin=" + pin)

        else:
            response = VoiceResponse()
            response.say("Invalid PIN. Please try again.")
            gather = Gather(num_digits=6, action="/enter_pin")
            response.append(gather)
        return PlainTextResponse(str(response), media_type="application/xml")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        response = VoiceResponse()
        response.say("An error occurred. Please try again later.")
        return PlainTextResponse(str(response), media_type="application/xml")


@app.post("/connect_call")
async def connect_call(request: Request):
    bot_details = request.query_params.get("bot_details")
    # Parse the bot_details if needed

    # log the bot details
    logger.info(f"Bot details: {bot_details}")

    response = VoiceResponse()
    response.say("Connecting to WebSocket.")
    stream_url = os.getenv("APPLICATION_BASE_URL")
    xml_content = open("templates/streams.xml").read()
    xml_content_with_url = xml_content.replace("{{ base_url }}", stream_url)
    return HTMLResponse(content=xml_content_with_url, media_type="application/xml")


# Exotel mapping for taking the pin input
@app.get("/ws_connect")
def ws_connect_exotel(request: Request):
    # CallSid=9953099079d70b89bbc1aa43ae6d1893&CallFrom=09901320691&CallTo=08069891522&Direction=incoming&Created=Tue%2C+03+Sep+2024+11%3A44%3A54&DialWhomNumber=&HangupLatencyStartTimeExocc=&HangupLatencyStartTime=&passthru=%22https%3A%5C%2F%5C%2Fsansadhak-response.reverieinc.com%5C%2Fexotel%5C%2Fdtmf-input%3FCallSid%3D9953099079d70b89bbc1aa43ae6d1893%26CallFrom%3D09901320691%26CallTo%3D08069891522%26Direction%3Dincoming%26Created%3DTue%252C%2B03%2BSep%2B2024%2B11%253A44%253A54%26DialCallDuration%3D0%26StartTime%3D2024-09-03%2B11%253A44%253A54%26EndTime%3D1970-01-01%2B05%253A30%253A00%26CallType%3Dcall-attempt%26DialWhomNumber%3D%26flow_id%3D11754%26tenant_id%3D1409%26From%3D09901320691%26To%3D08069891522%26CurrentTime%3D2024-09-03%2B11%253A44%253A59%26digits%3D%25228%2522%22&From=09901320691&To=08069891522&CurrentTime=2024-09-03+11%3A44%3A59
    params = request.query_params
    call_from = params["CallFrom"]
    logger.info(f"parameters in call {params}")
    number_details = {
        "07008567700": {"name": "Sidharth", "constituency": "Assandh"},
        "09547531359": {"name": "Suvojit", "constituency": "Karnal"},
        "09901320691": {"name": "Bhupen", "constituency": "Gurugram"},
        "09938899722": {"name": "Gourav", "constituency": "Kaithal"},
        "08104035237": {"name": "Hrusheekesh", "constituency": "Guhla"},
        "07735367840": {"name": "Rupesh", "constituency": "Panipat"},
        "09437986364": {"name": "Adarsh", "constituency": "Baroda"},
        "09818688082": {"name": "Anurag Saxena", "constituency": "Gurugram"},
        "08802737939": {"name": "Manish Arora", "constituency": "Bhiwani"},
        "09899543288": {"name": "Manish Joshi", "constituency": "Nuh"},
        "09310114007": {"name": "Kapil Yadav", "constituency": "Faridabad"},
        "07827561912": {"name": "Sushant Kumar", "constituency": "Ambala"},
        "09873737341": {"name": "Avanish Shahi", "constituency": "Panchkula"},
        "08882879931": {"name": "Srishti Johari", "constituency": "Narnaul"},
        "09897577655": {"name": "Rakesh Chowdhary", "constituency": "Kaithal"},
        "09845536275": {"name": "Mayuresh", "constituency": "Panipat"},
        "08095803061": {"name": "Sagar", "constituency": "Guhla"},
    }

    if call_from in number_details:
        _name = number_details[call_from]["name"]
        _constituency = number_details[call_from]["constituency"]
        return {
            "url": f"wss://ivr-api-dev.reverieinc.com/media?name={_name}&constituency={_constituency}"
        }

    return {
        "url": "wss://ivr-api-dev.reverieinc.com/media?name=bhupen&default_pin=215250"
    }


# Exotel mapping for taking the pin input
@app.get("/dtmf-input")
def dtmf_handler(request: Request, CallSid: str, digits: str):
    # /dtmf-input?CallSid=e12eb6e6fcd920019d7174b60099187v&CallFrom=09901320691&CallTo=08069891522&Direction=incoming&Created=Wed%2C+31+Jul+2024+16%3A42%3A56&DialCallDuration=0&StartTime=2024-07-31+16%3A42%3A56&EndTime=1970-01-01+05%3A30%3A00&CallType=call-attempt&DialWhomNumber=&flow_id=11754&tenant_id=1409&From=09901320691&To=08069891522&CurrentTime=2024-07-31+16%3A43%3A02&digits=%22215250%22
    # log_request(request)
    call_sid = CallSid
    pin = digits.strip('"')
    logger.info(f"call {call_sid} =(pin)=> {pin}")
    CALL_DETAILS[call_sid] = pin
    return {"status": "ok"}


# Exotel mapping for taking the websocket connection
@app.websocket("/media")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(call_data, flush=True)
    stream_sid = call_data["start"]["stream_sid"]

    # get callSid
    call_sid = call_data["start"]["call_sid"]
    print("WebSocket connection accepted")
    logger.debug(
        f"custom params from ivr {call_data['start'].get('custom_parameters',{})}"
    )
    # default_dtmf_pin = call_data["start"].get("custom_parameters",{}).get("default_pin", None)

    custom_parameters = call_data["start"].get("custom_parameters", {})
    pin = call_data["start"].get("custom_parameters", {}).get("pin", None)
    if not pin:
        pin = CALL_DETAILS[call_sid]
    logger.debug(f"call_sid {call_sid} | stream_sid {stream_sid}")
    bot_details = await get_bot_details_by_conversation_id(pin)

    if "constituency" in custom_parameters:
        await run_bot(
            websocket,
            stream_sid,
            call_sid,
            bot_details,
            custom_parameters=custom_parameters,
            provider="exotel"
        )
    else:
        custom_parameters = {"name": "Bhupen", "constituency": "Gurugram"}
        await run_bot(
            websocket,
            stream_sid,
            call_sid,
            bot_details,
            custom_parameters=custom_parameters,
            provider="exotel"
        )


@app.post("/connect_call_with_pin")
async def connect_call(request: Request):
    pin = request.query_params.get("pin")
    # Parse the bot_details if needed

    # log the bot details
    logger.info(f"pin: {pin}")

    response = VoiceResponse()
    response.say("Connecting to WebSocket.")
    stream_url = os.getenv("APPLICATION_BASE_URL")
    xml_content = open("templates/streams_with_pin.xml").read()
    xml_content_with_url = xml_content.replace("{{ base_url }}", stream_url)
    xml_content_with_url_with_pin = xml_content_with_url.replace("{{ pin }}", pin)

    return HTMLResponse(
        content=xml_content_with_url_with_pin, media_type="application/xml"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(call_data, flush=True)
    stream_sid = call_data["start"]["streamSid"]
    # get callSid
    call_sid = call_data["start"]["callSid"]
    print("WebSocket connection accepted")
    await run_bot(websocket, stream_sid, call_sid)


# make a websocket route which will be similar to the /ws route but will have a parameter for the conversation id
# and will return the bot details for that conversation id
@app.websocket("/ws/{pin}")
async def websocket_endpoint(websocket: WebSocket, pin: str):
    try:
        await websocket.accept()

        user_details = None
        bot_details = None
        
        bot_details_start_time = time.time()

        # log that we are getting data based on the conversation id
        logger.info(f"Getting bot details for conversation id: {pin}")
        bot_details = await get_bot_details_by_conversation_id(pin)

        # if bot_details is not found, then get the user details based on the pin
        if not bot_details:
            user_details = get_user_details_from_memory(pin)
            logger.debug(f"User details: {user_details}")

            # if user_details is found, get the conversation_id from user_details
            if user_details:
                conversation_id = user_details.get("conversation_id", "")

                # log conversation_id from user_details
                logger.debug(f"Conversation id from user details: {conversation_id}")

                get_bot_details_start_time = time.time()
                bot_details = await get_bot_details_by_conversation_id(conversation_id)
                logger.info("get bot details Time consuming: {:.4f}s".format(time.time() - get_bot_details_start_time))
                # logger.debug(f"Bot details: {bot_details}")

                # add the user details to the bot details
                name = user_details.get("name", "")
                constituency = user_details.get("constituency", "")
                recipient_phone_number = user_details.get("recipient_phone_number", "")
                bot_details["user_details"] = {
                    "name": name,
                    "constituency": constituency,
                    "recipient_phone_number": recipient_phone_number,
                    "conversation_id": conversation_id,
                }

        # log the bot details
        logger.debug(f"Bot details after modification: {bot_details}")
        
        logger.info("bot details Time consuming: {:.4f}s".format(time.time() - bot_details_start_time))

        start_data = websocket.iter_text()
        await start_data.__anext__()
        call_data = json.loads(await start_data.__anext__())
        logger.info(call_data)
        stream_sid = call_data["start"]["streamSid"]
        # get callSid
        call_sid = call_data["start"]["callSid"]
        logger.info("WebSocket connection accepted")
        await run_bot(websocket, stream_sid, call_sid, bot_details)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # Handle the error here


# api to make outbound call
@app.post("/bulk_call")
async def make_bulk_call(request: Request):
    try:
        call_details_list = await request.json()

        for call_details in call_details_list:

            # log the call details
            logger.info(f"Call details: {call_details}")

            conversation_id = call_details["conversation_id"]
            recipient_phone_number = call_details["recipient_phone_number"]
            # name = call_details["name"]
            # constituency = call_details["constituency"]
            name = call_details.get("name") 
            constituency = call_details.get("constituency")  # Returns None if missing

            # generate unique id for the user with uuid
            user_id_pin = str(uuid.uuid4())[:16]

            # save the user details in memory
            user_details = {
                "name": name,
                "constituency": constituency,
                "recipient_phone_number": recipient_phone_number,
                "conversation_id": conversation_id,
            }

            # call the function to store the user details
            store_user_details(user_id_pin, user_details)

            # call the function to make the call
            make_call(user_id_pin, recipient_phone_number)

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        logger.info(f"Exception occurred in make_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/exotel/bulk_call")
async def exotel_bulk_call(request: Request):
    try:
        call_details_list = await request.json()

        for call_details in call_details_list:

            # log the call details
            logger.info(f"Call details: {call_details}")

            conversation_id = call_details["conversation_id"]
            recipient_phone_number = call_details["recipient_phone_number"]
            name = call_details.get("name") 
            constituency = call_details.get("constituency")

            # generate unique id for the user with uuid
            user_id_pin = str(uuid.uuid4())[:16]

            # save the user details in memory
            user_details = {
                "name": name,
                # "constituency": constituency,
                "recipient_phone_number": recipient_phone_number,
                "conversation_id": conversation_id,
            }

            # call the function to store the user details
            store_user_details(user_id_pin, user_details)

            # call the function to make the call
            exotel_make_bulk_call(user_id_pin, recipient_phone_number)

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        logger.info(f"Exception occurred in exotel_bulk_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.post("/exotel_make_call")
async def exotel_make_calls(request: Request):
    try:
        call_details_list = await request.json()

        for call_details in call_details_list:

            # log the call details
            logger.info(f"Call details: {call_details}")

            conversation_id = call_details["conversation_id"]
            recipient_phone_number = call_details["recipient_phone_number"]
            name = call_details["name"]
            constituency = call_details.get("constituency")


            # generate unique id for the user with uuid
            user_id_pin = str(uuid.uuid4())[:16]

            # save the user details in memory
            user_details = {
                "name": name,
                # "constituency": constituency,
                "recipient_phone_number": recipient_phone_number,
                "conversation_id": conversation_id,
            }

            # call the function to store the user details
            store_user_details(user_id_pin, user_details)

            # call the function to make the call
            exotel_make_call(conversation_id, recipient_phone_number)

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        logger.info(f"Exception occurred in exotel_bulk_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
def launch_bot(room_url: str, token: str, bot_details: str):
    subprocess.Popen(["python", "simple_bot.py", room_url, token, bot_details])

# WebRTC implementation
@app.get("/start")
async def start_agent(request: Request,background_tasks: BackgroundTasks):

    print(f"!!! Creating room")
    # room = await daily_helpers["rest"].create_room(DailyRoomParams())
    room = daily_helpers["rest"].create_room(DailyRoomParams())
    print(room)
    print(f"!!! Room URL: {room.url}")
    # Ensure the room property is present
    if not room.url:
        raise HTTPException(
            status_code=500,
            detail="Missing 'room' property in request data. Cannot start agent without a target room!",
        )

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1
        for proc in bot_procs.values()
        if proc[1] == room.url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(
            status_code=500, detail=f"Max bot limited reach for room: {room.url}"
        )

    # Get the token for the room
    # token = await daily_helpers["rest"].get_token(room.url)
    token = daily_helpers["rest"].get_token(room.url)

    if not token:
        raise HTTPException(
            status_code=500, detail=f"Failed to get token for room: {room.url}"
        )

    # Spawn a new agent, and join the user session
    # Note: this is mostly for demonstration purposes (refer to 'deployment' in README)
    # conversation_id=659242 #ivr pin of the bot
    conversation_id = 862585
    bot_details = await get_bot_details_by_conversation_id(conversation_id)
    print(f"bot details: {bot_details}")
    # try:
    #     print(os.path.abspath(__file__))
    #     proc = subprocess.Popen(
    #         [
    #             # f"python3 -m simple_bot -u {room.url} -t {token}"
    #             f"python3 -m simple_bot -u {room.url} -b {bot_details}"
    #         ],
    #         shell=True,
    #         bufsize=1,
    #         cwd=os.path.dirname(os.path.abspath(__file__))
    #     )
    #     bot_procs[proc.pid] = (proc, room.url)
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=500, detail=f"Failed to start subprocess: {e}")

    # await simple_bot.main(room.url,token,bot_details)
    bot_details = str(bot_details)
    
    print(f"bot_details type {type(bot_details)}")
    print(f"room.url {room.url}")
    print(f"room.url type {type(room.url)}")
    print(f"token: {token}")
    print(f"token type: {type(token)}")

    # RedirectResponse(room.url)
    # subprocess.run(["python", "simple_bot.py", room.url, token, bot_details])
    background_tasks.add_task(launch_bot, room.url, token, bot_details)

    return RedirectResponse(url=room.url, status_code=303)
    # return "Room Created"


### Plivo implementation

#Plivo inbound
@app.post("/plivo_start_call_with_pin/")
# @app.api_route('/plivo_start_call_with_pin/', methods=['GET', 'POST'])

async def start_call(request: Request):
 
    WelcomeMessage = "Please enter your 6-digit PIN,."
    NoInput = "No input received. Goodbye."

    element = plivoxml.ResponseElement()

    base_url= os.getenv("APPLICATION_BASE_URL")
    response = (
        element.add(
            plivoxml.GetInputElement()
            .set_action(f"https://{base_url}/plivo_enter_pin")
            .set_method("POST")
            .set_input_type("dtmf")
            .set_digit_end_timeout(5)
            .set_redirect(True)
            .set_language("en-US")
            .add_speak(content=WelcomeMessage, voice="Polly.Salli", language="en-US")
        )
        .add_speak(content=NoInput)
        .to_string(False)
    )
    
    # Log the response (for debugging)
    print(response)
    
    # Return the XML response with the proper content type
    return Response(content=response, media_type="application/xml")



@app.post("/plivo_enter_pin")
async def enter_pin(Digits: str = Form(...), From: str = Form(...)):
    try:
        response = plivoxml.ResponseElement()
        pin = Digits
        from_number = From
        print(f"response: {response}")
        base_url= os.getenv("APPLICATION_BASE_URL")
        print(f"digit pressed: {pin}")

        pin = pin[:6]
        print(f"digit pressed: {pin}")

        if pin:
            print("inside if")
            response = plivoxml.ResponseElement()
            response.add(plivoxml.RedirectElement(f"https://{base_url}/plivo_connect_call_with_pin?pin={pin}"))
            print(response.to_string())
            return Response(content=response.to_string(), media_type="application/xml")



        else:
            print("inside else")
            response = plivoxml.ResponseElement()
            response.add_speak(content="Invalid PIN. Please try again.", voice="Polly.Salli", language="en-US")

    except Exception as e:

        print(f"error: {e}")



@app.post("/plivo_connect_call_with_pin")
async def connect_call(request: Request):
    pin = request.query_params.get("pin")
    # Parse the bot_details if needed

    # log the bot details
    logger.info(f"pin: {pin}")

    telephony_host = "https://" + os.getenv("APPLICATION_BASE_URL")
    bolna_host = "wss://" + os.getenv("APPLICATION_BASE_URL")

    pin = str(pin)
    response = plivoxml.ResponseElement()
    response.add(plivoxml.RedirectElement(f"{telephony_host}/plivo_connect/?websocket_host={bolna_host}&code={pin}"))
    print(response.to_string())
    return Response(content=response.to_string(), media_type="application/xml")



#plivo outbound
@app.post("/plivo_call")
async def make_call_plivo(request: Request):
# async def make_call_plivo():
    try:
        call_details_list = await request.json()

        for call_details in call_details_list:

            # log the call details
            logger.info(f"Call details: {call_details}")

            conversation_id = call_details["conversation_id"]
            recipient_phone_number = call_details["recipient_phone_number"]
        # telephony_host, bolna_host = populate_ngrok_tunnels()
        telephony_host = "https://" + os.getenv("APPLICATION_BASE_URL")
        bolna_host = "wss://" + os.getenv("APPLICATION_BASE_URL")
        agent_id = 3

        print(f"telephony_host: {telephony_host}")
        print(f"bolna_host: {bolna_host}")

        # adding hangup_url since plivo opens a 2nd websocket once the call is cut.
        # https://github.com/bolna-ai/bolna/issues/148#issuecomment-2127980509
        call = plivo_client.calls.create(
            from_=plivo_phone_number,
            to_=recipient_phone_number,
            answer_url=f"{telephony_host}/plivo_connect/?websocket_host={bolna_host}&code={conversation_id}",
            hangup_url=f"{telephony_host}/plivo_hangup_callback",
            answer_method="POST",
        )

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        print(f"Exception occurred in make_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/plivo_connect/")
async def plivo_connect(request: Request, websocket_host: str = Query(...),code: str = Query(...)):
    print("hello")
    print(code)
    print(websocket_host)
    try:
        _websocket_url = f"{websocket_host}/plivo/ws/{code}"

        response = """
        <Response>
            <Stream bidirectional="true" keepCallAlive="true">{}</Stream>
        </Response>
        """.format(
            _websocket_url
        )

        print(str(response))

        print(PlainTextResponse(str(response), status_code=200, media_type="text/xml"))
        return PlainTextResponse(str(response), status_code=200, media_type="text/xml")

    except Exception as e:
        print(f"Exception occurred in plivo_connect: {e}")


@app.post("/plivo_hangup_callback")
async def plivo_hangup_callback(request: Request):
    # add any post call hangup processing
    print("hangup")
    return PlainTextResponse("", status_code=200)


@app.websocket("/plivo/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    print("in ws")
    try:
        await websocket.accept()

        # log that we are getting data based on the conversation id
        logger.info(f"Getting bot details for conversation id: {conversation_id}")
        bot_details = await get_bot_details_by_conversation_id(conversation_id)
        logger.debug(f"Got Bot details: {bot_details}")

        start_data = websocket.iter_text()
        await start_data.__anext__()
        call_data = json.loads(await start_data.__anext__())
        print(call_data, flush=True)
        stream_sid = call_data["streamId"]

        call_sid = 1
        print("WebSocket connection accepted")
        await run_bot(websocket, stream_sid, call_sid, bot_details)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # Handle the error here


# Exotel mapping for taking the websocket connection
@app.websocket("/exotel/media")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(call_data, flush=True)
    stream_sid = call_data["start"]["stream_sid"]
    # get callSid
    call_sid = call_data["start"]["call_sid"]
    print("WebSocket connection accepted")

    pin = call_data["start"]["custom_parameters"].get("pin", "default_pin")
    
    # pin = CALL_DETAILS[call_sid]
    # bot_details = await get_bot_details_by_conversation_id(pin)
    
    user_details = None
    bot_details = None

    # log that we are getting data based on the conversation id
    logger.info(f"Getting bot details for conversation id: {pin}")
    bot_details = await get_bot_details_by_conversation_id(pin)

    # if bot_details is not found, then get the user details based on the pin
    if not bot_details:
        user_details = get_user_details_from_memory(pin)
        logger.debug(f"User details: {user_details}")

        # if user_details is found, get the conversation_id from user_details
        if user_details:
            conversation_id = user_details.get("conversation_id", "")

            # log conversation_id from user_details
            logger.debug(f"Conversation id from user details: {conversation_id}")

            bot_details = await get_bot_details_by_conversation_id(conversation_id)
            # logger.debug(f"Bot details: {bot_details}")

            # add the user details to the bot details
            name = user_details.get("name", "")
            constituency = user_details.get("constituency", "")
            recipient_phone_number = user_details.get("recipient_phone_number", "")
            bot_details["user_details"] = {
                "name": name,
                # "constituency": constituency,
                "recipient_phone_number": recipient_phone_number,
                "conversation_id": conversation_id,
            }

    # log the bot details
    logger.debug(f"Bot details after modification: {bot_details}")

    
    await run_bot(websocket, stream_sid, call_sid, bot_details)


@app.get("/exotel/ws_connect")
def ws_connect_exotel(request: Request):
    telephony_host = os.getenv("APPLICATION_BASE_URL")
    params = request.query_params

    # get the pin
    pin = params.get("pin", "default_pin")

    # log the params
    logger.info(f"parameters in call {params}")

    logger.info(f"telephony_host: {telephony_host}")

    return {"url": f"wss://{telephony_host}/exotel/media?name={pin}"}


def make_call(user_id_pin: str, recipient_phone_number: str):
    if not user_id_pin:
        raise HTTPException(status_code=404, detail="Pin not provided")

    if not recipient_phone_number:
        raise HTTPException(
            status_code=404, detail="Recipient phone number not provided"
        )

    stream_url = os.getenv("APPLICATION_BASE_URL")
    xml_content = open("templates/streams_with_pin.xml").read()
    xml_content_with_url = xml_content.replace("{{ base_url }}", stream_url)
    xml_content_with_url_with_pin = xml_content_with_url.replace("{{ pin }}", user_id_pin)

    try:
        call = twilio_client.calls.create(
            to=recipient_phone_number,
            from_=twilio_phone_number,
            twiml=xml_content_with_url_with_pin,
            method="POST",
            record=True,
        )
    except Exception as e:
        logger.info(f"make_call exception: {str(e)}")


def exotel_make_bulk_call(user_id_pin: str, recipient_phone_number: str):
    try:
        # log the user_id_pin and recipient_phone_number
        logger.info(f"User ID Pin: {user_id_pin}")
        logger.info(f"Recipient phone number: {recipient_phone_number}")

        if not user_id_pin:
            raise HTTPException(status_code=404, detail="Pin not provided")

        if not recipient_phone_number:
            raise HTTPException(
                status_code=404, detail="Recipient phone number not provided"
            )

        # if the recipient_phone_number is starting with +91 then make changes so that the number starts with 0
        if recipient_phone_number.startswith("+91"):
            recipient_phone_number = recipient_phone_number[3:]
            recipient_phone_number = "0" + recipient_phone_number

        # get the exotel environment variables
        exotel_sid = os.getenv("EXOTEL_SID")
        exotel_api_key = os.getenv("EXOTEL_API_KEY")
        exotel_api_token = os.getenv("EXOTEL_API_TOKEN")
        exotel_phone_number = os.getenv("EXOTEL_PHONE_NUMBER")

        # exotel_api_key = "your_exotel_api_key"
        exotel_api_url = f"https://{exotel_api_key}:{exotel_api_token}@api.in.exotel.com/v2/accounts/{exotel_sid}/campaigns"

        headers = {"Content-Type": "application/json"}

        data = {
            "campaigns": [
                {
                    "name": "chatbot_testing_15",
                    "caller_id": exotel_phone_number,
                    "from": [recipient_phone_number],
                    "url": f"http://my.exotel.com/{exotel_sid}/exoml/start_voice/14770",
                    "campaign_type": "static",
                    "retries": {
                        "number_of_retries": 1,
                        "interval_mins": 5,
                        "mechanism": "Exponential",
                        "on_status": ["failed", "busy", "no-answer"],
                    },
                    "custom_field": f"pin={user_id_pin}",
                }
            ]
        }

        # log that we are making request to Exotel API
        logger.info(f"Making request to Exotel API with data: {data}")

        response = requests.post(exotel_api_url, headers=headers, json=data)

        if response.status_code == 200:
            # log that Exotel call initiated successfully
            logger.info("Exotel call initiated successfully")

            return {"message": "Exotel call initiated successfully"}
        else:
            # log that Exotel call failed
            logger.error("Failed to initiate Exotel call")

            # log the response
            logger.error(f"Response from Exotel API: {response.json()}")

            return {"error": "Failed to initiate Exotel call"}

    except Exception as e:
        # log the exception
        logger.error(f"An error occurred: {str(e)}")
        # Handle the error here
        return {"error": "An error occurred"}
    
def exotel_make_call(user_id_pin: str, recipient_phone_number: str):
    try:
        # log the user_id_pin and recipient_phone_number
        logger.info(f"User ID Pin: {user_id_pin}")
        logger.info(f"Recipient phone number: {recipient_phone_number}")

        if not user_id_pin:
            raise HTTPException(status_code=404, detail="Pin not provided")

        if not recipient_phone_number:
            raise HTTPException(
                status_code=404, detail="Recipient phone number not provided"
            )

        # if the recipient_phone_number is starting with +91 then make changes so that the number starts with 0
        if recipient_phone_number.startswith("+91"):
            recipient_phone_number = recipient_phone_number[3:]
            recipient_phone_number = "0" + recipient_phone_number

        # get the exotel environment variables
        exotel_sid = os.getenv("EXOTEL_SID")
        exotel_api_key = os.getenv("EXOTEL_API_KEY")
        exotel_api_token = os.getenv("EXOTEL_API_TOKEN")
        exotel_phone_number = os.getenv("EXOTEL_PHONE_NUMBER")

        url = f"https://{exotel_api_key}:{exotel_api_token}@api.in.exotel.com/v1/Accounts/{exotel_sid}/Calls/connect"

        payload = f'From={recipient_phone_number}&CallerId={exotel_phone_number}&Url=http%3A%2F%2Fmy.exotel.com%2FExotel%2Fexoml%2Fstart_voice%2F15123&CustomField=pin%3D{user_id_pin}'
        headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        logger.info(url)
        logger.info(payload)
        print(response.text)

        if response.status_code == 200:
            # log that Exotel call initiated successfully
            logger.info("Exotel call initiated successfully")

            return {"message": "Exotel call initiated successfully"}
        else:
            # log that Exotel call failed
            logger.error("Failed to initiate Exotel call")

            # log the response
            logger.error(f"Response from Exotel API: {response.json()}")

            return {"error": "Failed to initiate Exotel call"}

    except Exception as e:
        # log the exception
        logger.error(f"An error occurred: {str(e)}")
        # Handle the error here
        return {"error": "An error occurred"}

@app.post("/clear_cache")
async def clear_cache():
    try:
        # Clear all keys in the Redis cache
        redis_client.flushdb()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/clear_cache/{conversation_id}")
async def clear_cache_by_conversation_id(conversation_id: str):
    try:
        # Check if the key exists in Redis
        if redis_client.exists(conversation_id):
            # Delete the key from Redis
            redis_client.delete(conversation_id)
            return {
                "message": f"Cache for conversation id {conversation_id} cleared successfully"
            }
        else:
            return {
                "message": f"No cache found for conversation id {conversation_id}"
            }
    except Exception as e:
        return {"error": str(e)}

@app.post("/clear_tts_cache")
async def clear_tts_cache():
    try:
        # Clear all keys in the Redis cache related to TTS
        keys = redis_client.keys("tts_cache:*")
        if keys:
            redis_client.delete(*keys)
            return {"message": "TTS cache cleared successfully"}
        else:
            return {"message": "No TTS cache found"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8766))
    uvicorn.run(app, host="0.0.0.0", port=port)
