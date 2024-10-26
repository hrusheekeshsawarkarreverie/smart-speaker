
import base64
import json

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.audio import ulaw_8000_to_pcm_16000, pcm_16000_to_ulaw_8000,ulaw_8000_to_pcm_8000,pcm_16000_to_slin_8000,slin_8000_to_pcm_16000
from loguru import logger

class PlivoFrameSerializer(FrameSerializer):
    SERIALIZABLE_TYPES = {
        AudioRawFrame: "audio",
    }

    def __init__(self, stream_sid: str,serializer_stt_provider: str):
        self._stream_sid = stream_sid
        self._serializer_stt_provider = serializer_stt_provider
        self.last_media_received = 0
        self.message_count = 0
        self.buffer =[]
        logger.debug(f"_serializer_stt_provider: {self._serializer_stt_provider}")




    def serialize(self, frame: Frame) -> str | bytes | None:

        if not isinstance(frame, AudioRawFrame):
            return None

        data = frame.audio

        # log that it is coming here
        # print("coming here in plivo serializer")

        # serialized_data = pcm_16000_to_ulaw_8000(data)
        serialized_data = pcm_16000_to_slin_8000(data)
        payload = base64.b64encode(serialized_data).decode("utf-8")
        # payload = base64.b64encode(data).decode("utf-8")
        # print(f"serialised payload {payload }")
        answer = {
            "event": "playAudio",
            # "streamSid": self._stream_sid,
            "media": {
                "payload": payload,
                'sampleRate': '8000',
                'contentType': 'audio/x-l16'
            }
        }

        # logger.debug("serialize plivo")
        return json.dumps(answer)

    def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)
        # buffer=[]
        # print("coming here in plivo deserializer")
        # logger.debug("deserialize plivo")
        if message["event"] != "media":
            return None
        else:
            # print(f"message: {message}")
            # payload_base64 = message["media"]["payload"]
            # payload = base64.b64decode(payload_base64)
            # print(f"payload_base64: {payload_base64}")

            # logger.debug(f"deserialised payload: {message}")
            # deserialized_data = ulaw_8000_to_pcm_16000(payload)
            # audio_frame = AudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=16000)
            # return audio_frame
            packet = message
            media_data = packet['media']
            media_audio = base64.b64decode(media_data['payload'])
            media_ts = int(media_data["timestamp"])
            # logger.debug(f"_serializer_stt_provider: {self._serializer_stt_provider}")
            if self._serializer_stt_provider == "reverie":
                # logger.debug("in reverie")
                # deserialized_data = ulaw_8000_to_pcm_8000(media_audio)
                deserialized_data = media_audio
                audio_frame = AudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=8000)
            else:
                # logger.debug("in other")
                deserialized_data = slin_8000_to_pcm_16000(media_audio)
            
            # print(f"audio: {media_audio}")
            # self.buffer.append(media_audio)
            audio_frame = AudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=16000)
            # print(f"audio_frame: {self.buffer}")

            return audio_frame

