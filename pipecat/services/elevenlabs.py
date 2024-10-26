
import aiohttp
import os
from typing import AsyncGenerator
from pipecat.frames.frames import AudioRawFrame, ErrorFrame, Frame
from pipecat.services.ai_services import TTSService
from loguru import logger
import redis


class ElevenLabsTTSService(TTSService):

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        api_key: str,
        voice_id: str,
        model: str = "eleven_turbo_v2",
        **kwargs,
        ):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._voice_id = voice_id
        self._aiohttp_session = aiohttp_session
        self._model = model

        # Initialize Redis client
        redis_host = os.getenv('REDIS_HOST', 'redis')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

    def can_generate_metrics(self) -> bool:
        return True

    def generate_cache_key(self, text: str) -> str:
        # Add a unique identifier to the cache key to avoid conflicts
        unique_id = str(hash(text))
        cache_key = f"tts_cache:{unique_id}_{self._voice_id}"
        return cache_key

    # write a function to save the audio data in the cache
    def save_audio_data(self, text, chunk):
        try:
            # get the cache key
            cache_key = self.generate_cache_key(f"{text}_{self._voice_id}")

            # append the new chunk to the existing data in Redis
            self.redis_client.append(cache_key, chunk)
        except Exception as e:
            logger.error(f"Error saving audio data: {e}")

    # write a function to get the audio data from the cache
    def get_audio_data(self, text):
        try:
            # get the cache key
            cache_key = self.generate_cache_key(f"{text}_{self._voice_id}")

            # fetch the audio data from Redis
            audio_data = self.redis_client.get(cache_key)
            if audio_data:
                logger.debug(f"Audio data fetched from cache: {cache_key}")
                return audio_data
            return None
        except Exception as e:
            logger.error(f"Error fetching audio data: {e}")
            return None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        text = text.replace("!", ".")
        text = text.replace("?", ".")
        
        logger.debug(f"Generating TTS: [{text}]")

        # check if the audio data is cached
        audio_data = self.get_audio_data(text)
        
        if audio_data:
            # log that audio data found in cache
            logger.debug(f"Audio data found in cache for text: {text}")
            await self.start_ttfb_metrics()
            for i in range(0, len(audio_data), 16000):
                await self.stop_ttfb_metrics()
                chunk = audio_data[i:i+16000]
                frame = AudioRawFrame(chunk, 16000, 1)
                yield frame
        else:
            logger.debug(f"Generating audio using ElevenLabs for text: {text}")
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"

            payload = {"text": text, "model_id": self._model}

            querystring = {"output_format": "pcm_16000", "optimize_streaming_latency": 5}

            headers = {
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
            }

            await self.start_ttfb_metrics()

            async with self._aiohttp_session.post(
                url, json=payload, headers=headers, params=querystring
            ) as r:
                if r.status != 200:
                    text = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status}, error: {text})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status}, error: {text})"
                    )
                    return

                async for chunk in r.content:
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = AudioRawFrame(chunk, 16000, 1)
                        yield frame
                        
                        # save the audio data in the cache 
                        self.save_audio_data(text, chunk)
    
    # async def generate_audio(self, text: str) -> bytes:
    #     logger.debug(f"Generating audio: [{text}]")

    #     # check if the audio data is cached
    #     audio_data = self.get_audio_data(text)
        
    #     if audio_data:
    #         # log that audio data found in cache
    #         logger.debug(f"Audio data found in cache for text: {text}")
    #         return audio_data
    #     else:
    #         url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"

    #         payload = {"text": text, "model_id": self._model}

    #         querystring = {"output_format": "pcm_16000", "optimize_streaming_latency": 2}

    #         headers = {
    #             "xi-api-key": self._api_key,
    #             "Content-Type": "application/json",
    #         }

    #         await self.start_ttfb_metrics()

    #         async with self._aiohttp_session.post(
    #             url, json=payload, headers=headers, params=querystring
    #         ) as r:
    #             if r.status != 200:
    #                 text = await r.text()
    #                 logger.error(
    #                     f"{self} error getting audio (status: {r.status}, error: {text})"
    #                 )
    #                 return None

    #             audio_data = await r.read()
                
    #             # save the audio data in the cache 
    #             self.save_audio_data(text, audio_data)
                
    #             return audio_data

