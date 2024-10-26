
import base64
import json

from pipecat.frames.frames import AudioRawFrame, Frame, EndStreamFrame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.audio import slin_8000_to_pcm_16000, pcm_16000_to_slin_8000
from loguru import logger

# logger_sink = logger.bind(task="exotel_stream")
# logger.add("exotel_logs/exotel_{time}.log", filter=lambda record: "task" in record["extra"] and record["extra"]["task"] == "exotel_stream")

class ExotelFrameSerializer(FrameSerializer):
    SERIALIZABLE_TYPES = {
        AudioRawFrame: "audio",
        EndStreamFrame: "end_stream"
    }

    def __init__(self, stream_sid: str,serializer_stt_provider: str):
        self._stream_sid = stream_sid
        self._serializer_stt_provider = serializer_stt_provider

    def serialize(self, frame: Frame) -> str | bytes | None:

        if isinstance(frame, AudioRawFrame):
            data = frame.audio
            logger.trace(f"audio data received {len(data)}")
            serialized_data = pcm_16000_to_slin_8000(data)
            logger.trace(f"serialized audio data received {len(serialized_data)}")
            payload = base64.b64encode(serialized_data).decode("utf-8")
            logger.trace(f"serialize: Stream SID {self._stream_sid}, len: {len(payload)}, payload: {payload}")
            answer = {
                "event": "media",
                "stream_sid": self._stream_sid,
                "media": {
                    "payload": payload
                }
            }

            return json.dumps(answer)

        elif isinstance(frame, EndStreamFrame):
            answer = {
                "event": "stop",
                "stream_sid": self._stream_sid
            }

            return json.dumps(answer)
    
        else:
            print("coming here")
            return None

    def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)

        if message["event"] != "media":
            return None
        else:
            payload_base64 = message["media"]["payload"]
            logger.trace(f"deserialize: payload_base64: {len(payload_base64)}, message: {message}")

            payload = base64.b64decode(payload_base64)
            if self._serializer_stt_provider == "reverie":
                # deserialized_data = ulaw_8000_to_pcm_8000(payload)
                deserialized_data = payload
                audio_frame = AudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=8000)
            else:
                deserialized_data = slin_8000_to_pcm_16000(payload)
            audio_frame = AudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=16000)
            return audio_frame
