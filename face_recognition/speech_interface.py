"""
Speech recognition and text-to-speech interface for the Face Recognition Agent.

Supports multiple speech recognition backends:
- Google Speech Recognition (online, free)
- Whisper (offline, requires model download)
- Sphinx (offline, less accurate)
"""

import os
import tempfile
from typing import Optional
from enum import Enum

# Check for available speech libraries
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Warning: speech_recognition not available. Install with: pip install SpeechRecognition")

try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    if not EDGE_TTS_AVAILABLE:
        print("Warning: No TTS available. Install with: pip install edge-tts")

try:
    from pywhispercpp.model import Model as WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Info: whisper.cpp not available. Using Google Speech Recognition instead.")


class SpeechEngine(Enum):
    """Available speech recognition engines."""
    GOOGLE = "google"  # Google Speech Recognition (online)
    WHISPER = "whisper"  # OpenAI Whisper (offline)
    SPHINX = "sphinx"  # CMU Sphinx (offline)


class SpeechInterface:
    """
    Handles speech recognition and text-to-speech for the Face Recognition Agent.
    """

    def __init__(
        self,
        engine: SpeechEngine = SpeechEngine.GOOGLE,
        whisper_model: str = "medium",  # Better for multilingual
        whisper_language: str = "",  # Empty for auto-detect (multilingual)
        tts_enabled: bool = True,
        tts_voice: str = "en-US-GuyNeural",
        timeout: int = 5,
        phrase_time_limit: int = 10
    ):
        """
        Initialize the speech interface.

        Args:
            engine: Speech recognition engine to use
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            whisper_language: Language code (en, sv, etc.) or empty for auto-detect
            tts_enabled: Enable text-to-speech
            tts_voice: Voice for edge-tts (e.g., en-US-GuyNeural, en-GB-SoniaNeural)
            timeout: Seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for a phrase
        """
        self.engine = engine
        self.whisper_model_name = whisper_model
        self.whisper_language = whisper_language
        self.tts_enabled = tts_enabled
        self.tts_voice = tts_voice
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit

        # Initialize recognizer
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            # Use 16kHz for Whisper (native rate), higher for others
            sample_rate = 16000 if engine == SpeechEngine.WHISPER else None
            self.microphone = sr.Microphone(sample_rate=sample_rate)
        else:
            self.recognizer = None
            self.microphone = None

        # Initialize Whisper if selected
        self.whisper_model = None
        if engine == SpeechEngine.WHISPER and WHISPER_AVAILABLE:
            print(f"Loading whisper.cpp model '{whisper_model}'...")
            self.whisper_model = WhisperModel(whisper_model)
            print("✓ whisper.cpp model loaded (Metal accelerated on Mac)")

        # Initialize TTS - prefer edge-tts for natural voices
        self.use_edge_tts = tts_enabled and EDGE_TTS_AVAILABLE
        self.tts_engine = None
        if tts_enabled and not EDGE_TTS_AVAILABLE and TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
            except Exception as e:
                print(f"Warning: Could not initialize TTS: {e}")
                self.tts_engine = None

    def speak(self, text: str, blocking: bool = False):
        """
        Speak text using text-to-speech.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to finish before returning
        """
        if not self.tts_enabled:
            return

        if self.use_edge_tts:
            self._speak_edge_tts(text, blocking=blocking)
        elif self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")

    def _speak_edge_tts(self, text: str, blocking: bool = False):
        """Speak using edge-tts (natural Microsoft voices)."""
        import subprocess
        import threading
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name

            # Generate speech
            async def generate():
                communicate = edge_tts.Communicate(text, self.tts_voice)
                await communicate.save(temp_path)

            # Handle both running and new event loops
            try:
                loop = asyncio.get_running_loop()
                # Already in async context - create task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, generate()).result()
            except RuntimeError:
                # No running loop - use asyncio.run
                asyncio.run(generate())

            # Play audio in background thread to avoid blocking UI
            def play_audio():
                try:
                    if os.path.exists(temp_path):
                        # Use afplay on Mac, ffplay on Linux
                        import platform
                        if platform.system() == "Darwin":
                            subprocess.run(["afplay", temp_path], check=True)
                        else:
                            subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_path],
                                           check=True, capture_output=True)
                        os.remove(temp_path)
                except Exception as e:
                    print(f"Audio playback error: {e}")

            thread = threading.Thread(target=play_audio, daemon=True)
            thread.start()

            # Wait for audio to finish if blocking
            if blocking:
                thread.join()

        except Exception as e:
            print(f"Edge TTS error: {e}")

    def listen(self, prompt: str = "", speak_prompt: bool = True) -> Optional[str]:
        """
        Listen for speech and convert to text.

        Args:
            prompt: Optional prompt to display/speak
            speak_prompt: Whether to speak the prompt via TTS

        Returns:
            Transcribed text or None if failed
        """
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
            print("Speech recognition not available. Please type instead.")
            return None

        # Display and optionally speak prompt
        if prompt:
            print(f"\n{prompt}")
            if speak_prompt:
                self.speak(prompt, blocking=True)  # Wait for prompt to finish before listening

        print("🎤 Listening... (speak now)")

        try:
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit
                )

            print("🔄 Processing speech...")

            # Transcribe based on selected engine
            if self.engine == SpeechEngine.WHISPER and self.whisper_model:
                text = self._transcribe_with_whisper(audio)
            elif self.engine == SpeechEngine.SPHINX:
                text = self._transcribe_with_sphinx(audio)
            else:  # Default to Google
                text = self._transcribe_with_google(audio)

            if text:
                print(f"✓ Heard: {text}")
                return text
            else:
                print("✗ Could not understand speech")
                return None

        except sr.WaitTimeoutError:
            print("⏱ Timeout: No speech detected")
            return None
        except sr.UnknownValueError:
            print("✗ Could not understand speech")
            return None
        except sr.RequestError as e:
            print(f"✗ Recognition error: {e}")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def _transcribe_with_google(self, audio) -> Optional[str]:
        """Transcribe audio using Google Speech Recognition."""
        try:
            return self.recognizer.recognize_google(audio)
        except Exception as e:
            print(f"Google recognition error: {e}")
            return None

    def _transcribe_with_whisper(self, audio) -> Optional[str]:
        """Transcribe audio using whisper.cpp (fast, Metal accelerated on Mac)."""
        if not self.whisper_model:
            print("Whisper model not loaded")
            return None

        try:
            # Save audio to temporary WAV file at 16kHz (required by whisper.cpp)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with open(temp_path, "wb") as wav_file:
                    wav_file.write(audio.get_wav_data(convert_rate=16000))

            # Transcribe with whisper.cpp (empty language = auto-detect)
            if self.whisper_language:
                segments = self.whisper_model.transcribe(temp_path, language=self.whisper_language)
            else:
                segments = self.whisper_model.transcribe(temp_path)
            text = ' '.join([segment.text for segment in segments]).strip()

            # Clean up temp file
            os.remove(temp_path)

            # Filter out placeholder tokens (indicates no speech detected)
            placeholders = ['[BLANK_AUDIO]', '[Clinking]', '[Music]', '[Applause]',
                           '[Laughter]', '[Silence]', '[ Silence ]', '[INAUDIBLE]',
                           '[CLICK]', '[ Click ]', '[Click]',
                           '(speaking in foreign language)', '(Speaking in foreign language)',
                           '(inaudible)', '(Inaudible)']
            for ph in placeholders:
                text = text.replace(ph, '').strip()

            return text if text else None

        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return None

    def _transcribe_with_sphinx(self, audio) -> Optional[str]:
        """Transcribe audio using CMU Sphinx."""
        try:
            return self.recognizer.recognize_sphinx(audio)
        except Exception as e:
            print(f"Sphinx recognition error: {e}")
            return None

    def ask_question(self, question: str, allow_text_fallback: bool = True, confirm: bool = False) -> str:
        """
        Ask a question and get an answer via speech or text.

        Args:
            question: Question to ask
            allow_text_fallback: Allow typing if speech fails
            confirm: Ask user to confirm/correct the recognized text

        Returns:
            Answer text
        """
        # Try speech recognition
        answer = self.listen(prompt=question, speak_prompt=True)

        # Fall back to text input if needed
        if not answer and allow_text_fallback:
            print("(You can also type your answer)")
            print(f"{question} ", end='', flush=True)
            answer = input().strip()
        elif answer and confirm:
            # Optional: confirm and allow correction
            print(f"Heard: \"{answer}\"")
            correction = input("Press Enter if correct, or type correction: ").strip()
            if correction:
                answer = correction

        return answer if answer else ""

    def confirm(self, question: str, default: bool = True) -> bool:
        """
        Ask a yes/no question via speech or text.

        Args:
            question: Question to ask
            default: Default value if unclear

        Returns:
            True for yes, False for no
        """
        prompt = f"{question} (say 'yes' or 'no')"
        answer = self.ask_question(prompt, allow_text_fallback=True).lower()

        yes_words = ["yes", "yeah", "yep", "sure", "ok", "okay", "y"]
        no_words = ["no", "nope", "nah", "n"]

        if any(word in answer for word in yes_words):
            return True
        elif any(word in answer for word in no_words):
            return False
        else:
            return default


def test_speech_interface():
    """Test the speech interface."""
    print("\n" + "=" * 80)
    print("SPEECH INTERFACE TEST")
    print("=" * 80)

    if not SPEECH_RECOGNITION_AVAILABLE:
        print("\nSpeech recognition not available!")
        print("Install with: pip install SpeechRecognition pyaudio")
        return

    # Test with Google (default)
    print("\n1. Testing with Google Speech Recognition...")
    interface = SpeechInterface(engine=SpeechEngine.GOOGLE, tts_enabled=True)

    print("\nTest 1: Simple question")
    name = interface.ask_question("What is your name?")
    print(f"Result: {name}")

    print("\nTest 2: Yes/No question")
    likes_pizza = interface.confirm("Do you like pizza?")
    print(f"Result: {likes_pizza}")

    # Test with Whisper if available
    if WHISPER_AVAILABLE:
        print("\n2. Testing with Whisper...")
        interface_whisper = SpeechInterface(
            engine=SpeechEngine.WHISPER,
            whisper_model="base",
            tts_enabled=True
        )

        print("\nTest 3: Whisper transcription")
        hobby = interface_whisper.ask_question("What is your hobby?")
        print(f"Result: {hobby}")

    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_speech_interface()
