"""
Face Recognition Agent using webcam and DeepFace.

This agent can recognize people using a webcam, store their information,
and respond to queries from other agents over A2A protocol.
"""

import asyncio
import json
import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from PIL import Image
import httpx

# Optional DeepFace import - will be checked at runtime
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available. Face recognition will not work.")

# Optional speech recognition import
try:
    from speech_interface import SpeechInterface, SpeechEngine
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Info: Speech recognition not available. Will use text input only.")


@dataclass
class PersonInfo:
    """Information about a recognized person."""

    name: str
    photo_path: str
    encoding_path: str
    metadata: Dict[str, Any]
    first_seen: str
    last_seen: str


@dataclass
class A2AMessage:
    """Message format for A2A communication."""

    role: str  # 'user' or 'assistant'
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class A2AResponse:
    """Response format for A2A communication."""

    message: str
    agent_id: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class PersonDatabase:
    """
    Manages storage and retrieval of person information.
    """

    def __init__(self, data_dir: str = "people_data"):
        """
        Initialize the person database.

        Args:
            data_dir: Directory to store person data
        """
        self.data_dir = Path(data_dir)
        self.photos_dir = self.data_dir / "photos"
        self.encodings_dir = self.data_dir / "encodings"
        self.db_file = self.data_dir / "database.json"

        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.photos_dir.mkdir(exist_ok=True)
        self.encodings_dir.mkdir(exist_ok=True)

        # Load or initialize database
        self.people: Dict[str, PersonInfo] = {}
        self._load_database()

    def _load_database(self):
        """Load the person database from disk."""
        if self.db_file.exists():
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                for name, info in data.items():
                    self.people[name] = PersonInfo(**info)

    def _save_database(self):
        """Save the person database to disk."""
        data = {name: asdict(person) for name, person in self.people.items()}
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_person(
        self,
        name: str,
        photo: np.ndarray,
        encoding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PersonInfo:
        """
        Add a new person to the database.

        Args:
            name: Person's name
            photo: Photo as numpy array (BGR format)
            encoding: Face encoding as numpy array
            metadata: Additional information about the person

        Returns:
            PersonInfo object
        """
        timestamp = datetime.utcnow().isoformat()

        # Save photo
        photo_filename = f"{name.replace(' ', '_')}_{timestamp.replace(':', '-')}.jpg"
        photo_path = self.photos_dir / photo_filename
        cv2.imwrite(str(photo_path), photo)

        # Save encoding if provided
        encoding_path = None
        if encoding is not None:
            encoding_filename = f"{name.replace(' ', '_')}_{timestamp.replace(':', '-')}.npy"
            encoding_path = self.encodings_dir / encoding_filename
            np.save(str(encoding_path), encoding)

        # Create person info
        person_info = PersonInfo(
            name=name,
            photo_path=str(photo_path),
            encoding_path=str(encoding_path) if encoding_path else "",
            metadata=metadata or {},
            first_seen=timestamp,
            last_seen=timestamp
        )

        self.people[name] = person_info
        self._save_database()

        return person_info

    def update_person(self, name: str, metadata: Dict[str, Any]):
        """
        Update person's metadata.

        Args:
            name: Person's name
            metadata: New metadata to merge
        """
        if name in self.people:
            self.people[name].metadata.update(metadata)
            self.people[name].last_seen = datetime.utcnow().isoformat()
            self._save_database()

    def get_person(self, name: str) -> Optional[PersonInfo]:
        """
        Get person information by name.

        Args:
            name: Person's name

        Returns:
            PersonInfo object or None if not found
        """
        return self.people.get(name)

    def list_people(self) -> List[str]:
        """
        Get list of all known people.

        Returns:
            List of person names
        """
        return list(self.people.keys())

    def search_by_metadata(self, query: Dict[str, Any]) -> List[PersonInfo]:
        """
        Search people by metadata fields.

        Args:
            query: Dictionary of field-value pairs to match

        Returns:
            List of matching PersonInfo objects
        """
        results = []
        for person in self.people.values():
            match = True
            for key, value in query.items():
                if key not in person.metadata or person.metadata[key] != value:
                    match = False
                    break
            if match:
                results.append(person)
        return results


class FaceRecognitionAgent:
    """
    Face recognition agent that can identify people via webcam and respond to A2A queries.
    """

    def __init__(
        self,
        agent_id: str = "face-recognition-agent",
        model_name: str = "gemma3:latest",
        base_url: str = "http://localhost:11434",
        data_dir: str = "people_data",
        face_model: str = "Facenet512",
        detector_backend: str = "opencv",
        speech_enabled: bool = True,
        speech_engine: str = "google",
        tts_enabled: bool = True,
        conversation_enabled: bool = True,
        conversation_questions: Optional[List[str]] = None
    ):
        """
        Initialize the Face Recognition Agent.

        Args:
            agent_id: Unique identifier for this agent
            model_name: Name of the Ollama model for natural language tasks
            base_url: Base URL for the Ollama API
            data_dir: Directory to store person data
            face_model: DeepFace model to use (Facenet512, VGG-Face, ArcFace, etc.)
            detector_backend: Face detector backend (opencv, ssd, dlib, mtcnn, retinaface)
            speech_enabled: Enable speech recognition for input
            speech_engine: Speech engine to use (google, whisper, sphinx)
            tts_enabled: Enable text-to-speech for prompts
            conversation_enabled: Enable conversation with recognized people
            conversation_questions: Questions to ask recognized people (e.g., ["favorite_candy", "interests"])
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.base_url = base_url
        self.face_model = face_model
        self.detector_backend = detector_backend
        self.speech_enabled = speech_enabled
        self.conversation_enabled = conversation_enabled

        # Initialize database
        self.db = PersonDatabase(data_dir)

        # Questions for new people (when saving)
        self.pending_questions: Set[str] = {"name"}  # Always ask for name

        # Questions for recognized people (during conversation)
        if conversation_questions is None:
            self.conversation_questions: List[str] = []
        else:
            self.conversation_questions = conversation_questions

        # Track last conversation time to avoid repeating too often
        self.last_conversation: Dict[str, float] = {}
        self.conversation_cooldown: int = 300  # 5 minutes between conversations

        # Camera
        self.camera = None

        # Initialize speech interface
        self.speech_interface = None
        if speech_enabled and SPEECH_AVAILABLE:
            try:
                engine_map = {
                    "google": SpeechEngine.GOOGLE,
                    "whisper": SpeechEngine.WHISPER,
                    "sphinx": SpeechEngine.SPHINX
                }
                engine = engine_map.get(speech_engine.lower(), SpeechEngine.GOOGLE)
                self.speech_interface = SpeechInterface(
                    engine=engine,
                    tts_enabled=tts_enabled
                )
                print(f"✓ Speech recognition enabled (engine: {speech_engine})")
            except Exception as e:
                print(f"Warning: Could not initialize speech interface: {e}")
                self.speech_interface = None
        elif speech_enabled:
            print("Info: Speech recognition requested but not available.")

        # Check if DeepFace is available
        if not DEEPFACE_AVAILABLE:
            print("WARNING: DeepFace not available. Please install with: pip install deepface")

    def _init_camera(self):
        """Initialize the webcam."""
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open webcam")

    def _release_camera(self):
        """Release the webcam."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the webcam.

        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        self._init_camera()
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame.

        Args:
            frame: Frame as numpy array (BGR format)

        Returns:
            List of face detection results
        """
        if not DEEPFACE_AVAILABLE:
            return []

        try:
            # DeepFace.extract_faces returns list of face objects
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []

    def recognize_face(self, face_img: np.ndarray) -> Optional[str]:
        """
        Recognize a face against known people in the database.

        Args:
            face_img: Face image as numpy array

        Returns:
            Name of recognized person or None if unknown
        """
        if not DEEPFACE_AVAILABLE or len(self.db.people) == 0:
            return None

        try:
            # Get embeddings for the input face
            input_embedding = DeepFace.represent(
                img_path=face_img,
                model_name=self.face_model,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )

            if not input_embedding:
                return None

            input_embedding = np.array(input_embedding[0]["embedding"])

            # Compare with known faces
            best_match = None
            best_distance = float('inf')

            # Model-specific thresholds for cosine distance
            # These are more reliable than Euclidean distance
            thresholds = {
                "VGG-Face": 0.40,
                "Facenet": 0.40,
                "Facenet512": 0.30,
                "OpenFace": 0.10,
                "DeepFace": 0.23,
                "DeepID": 0.015,
                "ArcFace": 0.68,
                "Dlib": 0.07,
                "SFace": 0.593
            }
            threshold = thresholds.get(self.face_model, 0.40)

            for name, person in self.db.people.items():
                if not person.encoding_path or not os.path.exists(person.encoding_path):
                    continue

                known_embedding = np.load(person.encoding_path)

                # Use cosine distance (more reliable than Euclidean)
                # Cosine distance = 1 - cosine similarity
                dot_product = np.dot(input_embedding, known_embedding)
                norm_product = np.linalg.norm(input_embedding) * np.linalg.norm(known_embedding)
                cosine_similarity = dot_product / norm_product
                distance = 1 - cosine_similarity

                print(f"  Comparing with {name}: distance={distance:.4f}, threshold={threshold}")

                if distance < best_distance and distance < threshold:
                    best_distance = distance
                    best_match = name

            if best_match:
                print(f"  ✓ Matched: {best_match} (distance: {best_distance:.4f})")
            else:
                print(f"  ✗ No match found (best distance: {best_distance:.4f})")

            return best_match

        except Exception as e:
            print(f"Error recognizing face: {e}")
            return None

    def get_face_encoding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face encoding for storage.

        Args:
            face_img: Face image as numpy array

        Returns:
            Face encoding as numpy array or None if failed
        """
        if not DEEPFACE_AVAILABLE:
            return None

        # Validate face image
        if face_img is None or face_img.size == 0:
            print("Invalid face image")
            return None

        # Ensure minimum size
        if face_img.shape[0] < 20 or face_img.shape[1] < 20:
            print(f"Face image too small: {face_img.shape}")
            return None

        try:
            embedding = DeepFace.represent(
                img_path=face_img,
                model_name=self.face_model,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )

            if embedding:
                return np.array(embedding[0]["embedding"])
            return None

        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None

    async def ask_llm(self, question: str, context: str = "") -> str:
        """
        Ask the LLM a question using Ollama.

        Args:
            question: Question to ask
            context: Additional context

        Returns:
            LLM response
        """
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": question})

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.model_name, "messages": messages, "stream": False},
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "")
            except Exception as e:
                return f"Error: {str(e)}"

    async def collect_person_info(self, name: str, frame: np.ndarray) -> Dict[str, Any]:
        """
        Collect information about a person through interactive prompts.
        Uses speech recognition if enabled, otherwise falls back to text input.

        Args:
            name: Person's name
            frame: Frame containing the person's face

        Returns:
            Dictionary of collected information
        """
        info = {"name": name}

        # Greet the new person
        greeting = f"Nice to meet you, {name}!"
        print(f"\n{greeting}")
        if self.speech_interface:
            self.speech_interface.speak(greeting)

        # Ask pending questions
        for question_key in self.pending_questions:
            if question_key == "name":
                continue  # Already have the name

            # Format question nicely
            formatted_question = question_key.replace("_", " ").capitalize()

            # Use speech or text input
            if self.speech_interface:
                answer = self.speech_interface.ask_question(
                    f"What is your {formatted_question}?",
                    allow_text_fallback=True
                )
            else:
                print(f"\n{formatted_question}: ", end='', flush=True)
                answer = input().strip()

            if answer:
                info[question_key] = answer

        # Thank them
        thanks = "Thank you! I'll remember you."
        print(thanks)
        if self.speech_interface:
            self.speech_interface.speak(thanks)

        return info

    async def have_conversation(self, name: str) -> Dict[str, Any]:
        """
        Have a conversation with a recognized person.

        Args:
            name: Person's name

        Returns:
            Dictionary of new information collected
        """
        import time

        # Check cooldown
        current_time = time.time()
        if name in self.last_conversation:
            time_since_last = current_time - self.last_conversation[name]
            if time_since_last < self.conversation_cooldown:
                print(f"(Cooldown: talked with {name} {int(time_since_last)}s ago)")
                return {}

        # Greet the person
        greeting = f"Hello {name}! Nice to see you again!"
        print(f"\n{'=' * 60}")
        print(greeting)
        if self.speech_interface:
            self.speech_interface.speak(greeting)

        new_info = {}

        # Ask conversation questions
        for question_key in self.conversation_questions:
            # Check if we already have this info
            person = self.db.get_person(name)
            if person and question_key in person.metadata:
                continue  # Skip if we already know

            # Format question nicely
            formatted_question = question_key.replace("_", " ").capitalize()

            # Ask the question
            if self.speech_interface:
                answer = self.speech_interface.ask_question(
                    f"Tell me, what is your {formatted_question}?",
                    allow_text_fallback=True
                )
            else:
                print(f"{formatted_question}: ", end='', flush=True)
                answer = input().strip()

            if answer:
                new_info[question_key] = answer

        # Update database if we got new info
        if new_info:
            self.db.update_person(name, new_info)
            print(f"✓ Updated information for {name}")

        # Update last conversation time
        self.last_conversation[name] = current_time

        # Say goodbye
        goodbye = "Thanks for chatting!"
        if self.speech_interface:
            self.speech_interface.speak(goodbye)
        print(goodbye)
        print("=" * 60 + "\n")

        return new_info

    async def process_recognition_loop(self, duration: int = 30):
        """
        Run the face recognition loop for a specified duration.

        Args:
            duration: How long to run in seconds (0 for continuous)
        """
        print("\n" + "=" * 80)
        print("FACE RECOGNITION AGENT - STARTING")
        print("=" * 80)
        print("\nPress 'q' to quit, 's' to save current frame")
        print("The agent will recognize faces and collect information.\n")

        self._init_camera()
        start_time = datetime.now()

        try:
            while True:
                # Check duration
                if duration > 0 and (datetime.now() - start_time).seconds >= duration:
                    break

                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    print("Failed to capture frame")
                    await asyncio.sleep(1)
                    continue

                # Detect faces
                faces = self.detect_faces(frame)

                # Draw rectangles around faces
                display_frame = frame.copy()
                for face_obj in faces:
                    facial_area = face_obj.get("facial_area", {})
                    x = facial_area.get("x", 0)
                    y = facial_area.get("y", 0)
                    w = facial_area.get("w", 0)
                    h = facial_area.get("h", 0)

                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]

                    # Try to recognize
                    recognized_name = self.recognize_face(face_img)

                    # Draw rectangle and label
                    color = (0, 255, 0) if recognized_name else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)

                    label = recognized_name if recognized_name else "Unknown"
                    cv2.putText(
                        display_frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                    )

                    # Have conversation with recognized person
                    if recognized_name and self.conversation_enabled and self.conversation_questions:
                        # Trigger conversation (will check cooldown internally)
                        await self.have_conversation(recognized_name)

                    # If unknown face detected, offer to add
                    if not recognized_name and len(faces) == 1:  # Only one face
                        cv2.imshow('Face Recognition', display_frame)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('s'):  # Save this person
                            print("\n" + "-" * 80)
                            print("New person detected!")

                            # Get name via speech or text
                            if self.speech_interface:
                                name = self.speech_interface.ask_question(
                                    "What is your name?",
                                    allow_text_fallback=True
                                )
                            else:
                                print("Please provide the person's name: ", end='', flush=True)
                                name = input().strip()

                            if name:
                                # Get face encoding
                                encoding = self.get_face_encoding(face_img)

                                # Collect additional information
                                metadata = await self.collect_person_info(name, frame)

                                # Add to database
                                person_info = self.db.add_person(
                                    name=name,
                                    photo=face_img,
                                    encoding=encoding,
                                    metadata=metadata
                                )

                                print(f"\nAdded {name} to the database!")
                                print(f"Photo saved to: {person_info.photo_path}")
                                print("-" * 80 + "\n")

                                # Reset pending questions after successful add
                                self.pending_questions = {"name"}

                        elif key == ord('q'):
                            break

                        continue

                # Display frame
                cv2.imshow('Face Recognition', display_frame)

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                await asyncio.sleep(0.1)

        finally:
            self._release_camera()
            cv2.destroyAllWindows()
            print("\nFace recognition loop ended.")

    async def handle_a2a_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an A2A protocol request from another agent.

        Supported operations:
        - query_person: Get information about a person by name
        - list_people: Get list of all known people
        - request_questions: Add questions to ask when meeting new people
        - search_people: Search people by metadata

        Args:
            request_data: The A2A request data

        Returns:
            A2A response data
        """
        operation = request_data.get("operation", "")

        if operation == "query_person":
            name = request_data.get("name", "")
            person = self.db.get_person(name)

            if person:
                return {
                    "status": "success",
                    "person": {
                        "name": person.name,
                        "metadata": person.metadata,
                        "first_seen": person.first_seen,
                        "last_seen": person.last_seen
                    },
                    "agent_info": {
                        "id": self.agent_id,
                        "type": "face_recognition_agent"
                    }
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"Person '{name}' not found in database",
                    "agent_info": {
                        "id": self.agent_id,
                        "type": "face_recognition_agent"
                    }
                }

        elif operation == "list_people":
            people = self.db.list_people()
            return {
                "status": "success",
                "people": people,
                "count": len(people),
                "agent_info": {
                    "id": self.agent_id,
                    "type": "face_recognition_agent"
                }
            }

        elif operation == "request_questions":
            questions = request_data.get("questions", [])
            for question in questions:
                self.pending_questions.add(question)

            return {
                "status": "success",
                "message": f"Added {len(questions)} questions to collection queue",
                "pending_questions": list(self.pending_questions),
                "agent_info": {
                    "id": self.agent_id,
                    "type": "face_recognition_agent"
                }
            }

        elif operation == "set_conversation_questions":
            questions = request_data.get("questions", [])
            replace = request_data.get("replace", False)

            if replace:
                self.conversation_questions = questions
            else:
                # Add to existing questions
                for q in questions:
                    if q not in self.conversation_questions:
                        self.conversation_questions.append(q)

            return {
                "status": "success",
                "message": f"Conversation questions {'replaced with' if replace else 'updated to include'} {len(questions)} questions",
                "conversation_questions": self.conversation_questions,
                "agent_info": {
                    "id": self.agent_id,
                    "type": "face_recognition_agent"
                }
            }

        elif operation == "get_conversation_config":
            return {
                "status": "success",
                "conversation_enabled": self.conversation_enabled,
                "conversation_questions": self.conversation_questions,
                "conversation_cooldown": self.conversation_cooldown,
                "agent_info": {
                    "id": self.agent_id,
                    "type": "face_recognition_agent"
                }
            }

        elif operation == "set_conversation_config":
            if "enabled" in request_data:
                self.conversation_enabled = request_data["enabled"]
            if "cooldown" in request_data:
                self.conversation_cooldown = request_data["cooldown"]

            return {
                "status": "success",
                "message": "Configuration updated",
                "conversation_enabled": self.conversation_enabled,
                "conversation_cooldown": self.conversation_cooldown,
                "agent_info": {
                    "id": self.agent_id,
                    "type": "face_recognition_agent"
                }
            }

        elif operation == "search_people":
            query = request_data.get("query", {})
            results = self.db.search_by_metadata(query)

            return {
                "status": "success",
                "results": [
                    {
                        "name": person.name,
                        "metadata": person.metadata
                    }
                    for person in results
                ],
                "count": len(results),
                "agent_info": {
                    "id": self.agent_id,
                    "type": "face_recognition_agent"
                }
            }

        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "supported_operations": [
                    "query_person",
                    "list_people",
                    "request_questions",
                    "search_people",
                    "set_conversation_questions",
                    "get_conversation_config",
                    "set_conversation_config"
                ],
                "agent_info": {
                    "id": self.agent_id,
                    "type": "face_recognition_agent"
                }
            }


async def main():
    """Main entry point - starts face recognition with conversation mode."""
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition Agent")
    parser.add_argument("--no-speech", action="store_true", help="Disable speech (text only)")
    parser.add_argument("--menu", action="store_true", help="Show menu instead of direct start")
    parser.add_argument("--questions", nargs="+", default=["favorite_candy", "interests", "hobby"],
                        help="Questions to ask (default: favorite_candy interests hobby)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FACE RECOGNITION AGENT")
    print("=" * 80)

    # Create agent with default questions for both new people and conversations
    agent = FaceRecognitionAgent(
        speech_enabled=not args.no_speech,
        conversation_enabled=True,
        conversation_questions=args.questions
    )

    # Also set pending questions for new people
    for q in args.questions:
        agent.pending_questions.add(q)

    print(f"\n✓ Speech: {'Enabled' if agent.speech_interface else 'Disabled'}")
    print(f"✓ Questions to ask: {', '.join(args.questions)}")

    # Go directly to face recognition or show menu
    if not args.menu:
        print("\nStarting face recognition...")
        print("Press 's' to save a new person, 'q' to quit")
        await agent.process_recognition_loop(duration=0)
        return

    # Show menu if requested
    print("\nOptions:")
    print("  1. Run face recognition loop")
    print("  2. List known people")
    print("  3. Query person information")
    print("  4. Test A2A requests")
    print("  5. Exit")

    while True:
        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            duration = input("Duration in seconds (0 for continuous): ").strip()
            duration = int(duration) if duration.isdigit() else 30
            await agent.process_recognition_loop(duration=duration)

        elif choice == "2":
            people = agent.db.list_people()
            print(f"\nKnown people ({len(people)}):")
            for name in people:
                print(f"  - {name}")

        elif choice == "3":
            name = input("Enter person's name: ").strip()
            person = agent.db.get_person(name)
            if person:
                print(f"\nPerson: {person.name}")
                print(f"First seen: {person.first_seen}")
                print(f"Last seen: {person.last_seen}")
                print(f"Metadata: {json.dumps(person.metadata, indent=2)}")
            else:
                print(f"Person '{name}' not found")

        elif choice == "4":
            # Test A2A requests
            print("\nTest A2A Operations:")
            print("  a. List people")
            print("  b. Query person")
            print("  c. Request questions")
            print("  d. Search people")

            op_choice = input("Choose operation (a-d): ").strip().lower()

            if op_choice == "a":
                result = await agent.handle_a2a_request({"operation": "list_people"})
                print(f"\nResult: {json.dumps(result, indent=2)}")

            elif op_choice == "b":
                name = input("Enter person's name: ").strip()
                result = await agent.handle_a2a_request({
                    "operation": "query_person",
                    "name": name
                })
                print(f"\nResult: {json.dumps(result, indent=2)}")

            elif op_choice == "c":
                questions_input = input("Enter questions (comma-separated): ").strip()
                questions = [q.strip() for q in questions_input.split(",")]
                result = await agent.handle_a2a_request({
                    "operation": "request_questions",
                    "questions": questions
                })
                print(f"\nResult: {json.dumps(result, indent=2)}")

            elif op_choice == "d":
                key = input("Enter metadata key to search: ").strip()
                value = input("Enter value: ").strip()
                result = await agent.handle_a2a_request({
                    "operation": "search_people",
                    "query": {key: value}
                })
                print(f"\nResult: {json.dumps(result, indent=2)}")

        elif choice == "5":
            print("Exiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())
