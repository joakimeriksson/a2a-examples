"""Face Recognition Agent Package."""

from .face_recognition_agent import (
    FaceRecognitionAgent,
    PersonDatabase,
    PersonInfo,
    A2AMessage,
    A2AResponse,
)

try:
    from .speech_interface import SpeechInterface, SpeechEngine
    __all__ = [
        "FaceRecognitionAgent",
        "PersonDatabase",
        "PersonInfo",
        "A2AMessage",
        "A2AResponse",
        "SpeechInterface",
        "SpeechEngine",
    ]
except ImportError:
    __all__ = [
        "FaceRecognitionAgent",
        "PersonDatabase",
        "PersonInfo",
        "A2AMessage",
        "A2AResponse",
    ]
