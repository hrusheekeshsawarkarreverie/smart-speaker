
import base64
import json

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.audio import ulaw_8000_to_pcm_16000, pcm_16000_to_ulaw_8000,ulaw_8000_to_pcm_8000
from loguru import logger

class TwilioFrameSerializer(FrameSerializer):
    SERIALIZABLE_TYPES = {
        AudioRawFrame: "audio",
    }

    def __init__(self, stream_sid: str,serializer_stt_provider: str):
        self._stream_sid = stream_sid
        self._serializer_stt_provider = serializer_stt_provider

    def serialize(self, frame: Frame) -> str | bytes | None:
        if not isinstance(frame, AudioRawFrame):
            return None

        data = frame.audio

        serialized_data = pcm_16000_to_ulaw_8000(data)
        payload = base64.b64encode(serialized_data).decode("utf-8")
        answer = {
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {
                "payload": payload
            }
        }

        return json.dumps(answer)

    def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)

        if message["event"] != "media":
            return None
        else:
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            if self._serializer_stt_provider == "reverie":
                deserialized_data = ulaw_8000_to_pcm_8000(payload)
                audio_frame = AudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=8000)
            else:
                deserialized_data = ulaw_8000_to_pcm_16000(payload)
                audio_frame = AudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=16000)
            return audio_frame
