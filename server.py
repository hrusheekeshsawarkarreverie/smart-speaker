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


# parser = argparse.ArgumentParser(description="A script that runs different Setups based on arguments (WebRTC and IVR.")

# # Adding an optional argument
# parser.add_argument('--arg', type=str, help='An optional argument to trigger a specific function')

# # Parsing the arguments
# args = parser.parse_args()

# print(args)
# if args.arg:
#     print("Starting WebRTC Setup")
#     app = FastAPI(lifespan=lifespan)
# else:
#     print("Starting IVR Setup")
#     app = FastAPI()

# app = FastAPI()
app = FastAPI(lifespan=lifespan)

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
    token = daily_helpers["rest"].get_token(room.url,expiry_time=60*60)

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
