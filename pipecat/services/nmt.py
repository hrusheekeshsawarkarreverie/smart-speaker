import asyncio
import json
import requests
import aiohttp
import time
from typing import List
from requests import RequestException
import http.client
import json
from loguru import logger




class NMTService():
    def __init__(
        self,
        text: str,
        tgt_lan: str, 
        nmt_provider: str,
        src_lan: str="en", #en
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._text = text
        self._nmt_provider = nmt_provider
        self._tgt_lan=tgt_lan
        self._src_lan=src_lan

    async def translate(self):
        logger.debug(f"src: {self._src_lan}, tgt: {self._tgt_lan}, text: {self._text}: provider: {self._nmt_provider}")
        if self._nmt_provider == "Reverie":
            return await self.reverie_nmt()
        elif self._nmt_provider == "Google":
            return await self.google_nmt()

    async def google_nmt(self, wait_time=60):
        url = "https://translate.googleapis.com/translate_a/single"
        querystring = {"client": "gtx", "sl": self._src_lan, "tl": self._tgt_lan, "dt": "t", "q": self._text}
        headers = {'Cache-Control': "no-cache", 'authority': "translate.googleapis.com"}

        while True:
            try:
                response = requests.get(url, headers=headers, params=querystring)
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)
                
                if response.status_code == 429:
                    # API rate limit exceeded
                    print(f"API rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff (increase wait time for next retry)
                    continue

                translations = []
                for translation in response.json()[0]:
                    if translation[0] not in translations:
                        translations.append(translation[0])
                logger.debug(f"Google Translated Text: {translation[0]}")
                return translations[0]

            except RequestException as e:
                print(f"Request failed: {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff



    async def reverie_nmt(self):
        # print(f"src: {self._src_lan}, tgt: {tgt}, text: {self._text}")
        print(f" tgt: {self._tgt_lan}, text: {self._text}")


        host = "172.18.0.4"
        port = 8082
        endpoint = "/translate"

        # Prepare headers and data
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            'data': [self._text],  # Replace with your actual 'text'
            'mask': True,
            'src': self._src_lan,    # Replace with actual source language
            'tgt': self._tgt_lan,    # Replace with actual target language
            'domain': 0,
            'filter_profane': False
        }

        # Convert data to JSON
        json_data = json.dumps(data)

        # Create a connection to the server
        conn = http.client.HTTPConnection(host, port)

        try:
            # Send POST request
            conn.request("POST", endpoint, body=json_data, headers=headers)

            # Get the response
            response = conn.getresponse()

            # Read the response
            response_data = response.read().decode()

            # Print the response status and data
            print("Status:", response.status)
            print("Response:", response_data)

            response_json = json.loads(response_data)

            # Extract the translated text
            translated_text = response_json["result"][0][0]

            # Print the translated text
            # print("Translated Text:", translated_text)
            logger.debug(f"Reverie Translated Text: {translated_text}")
            return translated_text

        except Exception as e:
            translated_text = self._text
            logger.debug(f"Not Translated Text: {translated_text}")
            logger.debug(f"Exception: {e}")
            return translated_text

        finally:
            # Close the connection
            conn.close() 

    # res = reverie_nmt(self)
    