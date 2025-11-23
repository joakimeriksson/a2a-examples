"""
Face Recognition Agent using webcam and InsightFace.

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

# Optional InsightFace import - will be checked at runtime
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Face recognition will not work.")

# Optional speech recognition import
try:
    from speech_interface import SpeechInterface, SpeechEngine
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Info: Speech recognition not available. Will use text input only.")


def format_question(question_key: str) -> str:
    """Format a question key into a natural question string."""
    # Map of question keys to natural questions
    question_map = {
        "name": "What is your name?",
        "interests": "What are your interests?",
        "hobby": "What are your hobbies?",
        "hobbies": "What are your hobbies?",
        "favorite_candy": "What is your favorite candy?",
        "favorite_food": "What is your favorite food?",
        "favorite_color": "What is your favorite color?",
        "favorite_music": "What kind of music do you like?",
        "favorite_movie": "What is your favorite movie?",
        "occupation": "What do you do for work?",
        "age": "How old are you?",
    }

    if question_key in question_map:
        return question_map[question_key]

    # Default formatting for unknown keys
    formatted = question_key.replace("_", " ")
    return f"What is your {formatted}?"


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

    def add_embedding(self, name: str, encoding: np.ndarray) -> bool:
        """
        Add an additional face embedding for a person.

        This improves recognition by having multiple samples from different angles.

        Args:
            name: Person's name
            encoding: Face encoding as numpy array

        Returns:
            True if embedding was added, False if person not found
        """
        if name not in self.people:
            return False

        timestamp = datetime.utcnow().isoformat()
        encoding_filename = f"{name.replace(' ', '_')}_{timestamp.replace(':', '-')}.npy"
        encoding_path = self.encodings_dir / encoding_filename
        np.save(str(encoding_path), encoding)
        return True

    def get_all_embeddings(self, name: str) -> List[np.ndarray]:
        """
        Get all face embeddings for a person.

        Args:
            name: Person's name

        Returns:
            List of face embeddings
        """
        if name not in self.people:
            return []

        embeddings = []
        name_prefix = name.replace(' ', '_') + '_'

        # Get the primary embedding
        person = self.people[name]
        if person.encoding_path and os.path.exists(person.encoding_path):
            embeddings.append(np.load(person.encoding_path))

        # Find additional embeddings
        for encoding_file in self.encodings_dir.glob(f"{name_prefix}*.npy"):
            if str(encoding_file) != person.encoding_path:
                try:
                    embeddings.append(np.load(str(encoding_file)))
                except Exception:
                    pass

        return embeddings

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
        det_size: tuple = (640, 640),
        speech_enabled: bool = True,
        speech_engine: str = "whisper",
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
            det_size: Detection size for InsightFace (default: 640x640)
            speech_enabled: Enable speech recognition for input
            speech_engine: Speech engine to use (google, whisper, sphinx)
            tts_enabled: Enable text-to-speech for prompts
            conversation_enabled: Enable conversation with recognized people
            conversation_questions: Questions to ask recognized people (e.g., ["favorite_candy", "interests"])
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.base_url = base_url
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

        # Status for display
        self.status = "Initializing..."

        # Camera and current frame for UI updates
        self.camera = None
        self.current_frame = None

        # Initialize InsightFace
        self.face_app = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(
                    name='buffalo_l',  # Best accuracy model
                    providers=['CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=0, det_size=det_size)
                print(f"✓ InsightFace initialized (model: buffalo_l, det_size: {det_size})")
            except Exception as e:
                print(f"Warning: Could not initialize InsightFace: {e}")
                self.face_app = None
        else:
            print("WARNING: InsightFace not available. Please install with: pip install insightface onnxruntime")

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

    def _show_listening_indicator(self, frame: np.ndarray, message: str = "LISTENING..."):
        """Draw a prominent listening indicator on the frame and display it."""
        display = frame.copy()

        # Draw semi-transparent overlay at top
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 80), (0, 100, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Draw ear symbol and text
        cv2.putText(
            display, f"[EAR] {message}", (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
        )

        # Show the frame
        cv2.imshow('Face Recognition', display)
        cv2.waitKey(1)  # Force display update

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
        Detect faces in a frame using InsightFace.

        Args:
            frame: Frame as numpy array (BGR format)

        Returns:
            List of face detection results with embeddings
        """
        if self.face_app is None:
            return []

        try:
            # InsightFace returns face objects with bbox and embedding
            faces = self.face_app.get(frame)

            # Convert to consistent format
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)

                # Get gaze/attention info from pose (yaw, pitch, roll)
                looking_at_camera = False
                pose = getattr(face, 'pose', None)
                if pose is not None:
                    yaw, pitch, roll = pose
                    # Person is looking at camera if yaw and pitch are small
                    # Yaw: left/right rotation, Pitch: up/down rotation
                    looking_at_camera = abs(yaw) < 20 and abs(pitch) < 20

                results.append({
                    "facial_area": {
                        "x": int(bbox[0]),
                        "y": int(bbox[1]),
                        "w": int(bbox[2] - bbox[0]),
                        "h": int(bbox[3] - bbox[1])
                    },
                    "embedding": face.embedding,
                    "det_score": float(face.det_score),
                    "looking_at_camera": looking_at_camera,
                    "pose": pose.tolist() if pose is not None else None
                })
            return results
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []

    @staticmethod
    def _cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine distance between two embeddings.
        Returns a large number if either vector has zero norm.
        """
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return float("inf")
        cosine_similarity = np.dot(vec1, vec2) / norm_product
        return 1 - cosine_similarity

    def recognize_face_embedding(self, input_embedding: np.ndarray) -> Optional[str]:
        """
        Recognize a face against known people using pre-computed embedding.

        Args:
            input_embedding: Face embedding from InsightFace

        Returns:
            Name of recognized person or None if unknown
        """
        if len(self.db.people) == 0:
            return None

        try:
            # Compare with known faces
            best_match = None
            best_distance = float('inf')

            # InsightFace uses cosine similarity, threshold around 0.4-0.5 for cosine distance
            threshold = 0.45

            for name, person in self.db.people.items():
                # Get all embeddings for this person (supports multiple samples)
                embeddings = self.db.get_all_embeddings(name)
                if not embeddings:
                    continue

                # Find best match among all embeddings for this person
                person_best_distance = float('inf')
                for known_embedding in embeddings:
                    # Calculate cosine distance between embeddings for reliable matching
                    distance = self._cosine_distance(input_embedding, known_embedding)
                    if distance < person_best_distance:
                        person_best_distance = distance

                print(f"  Comparing with {name}: distance={person_best_distance:.4f}, threshold={threshold}")

                if person_best_distance < best_distance and person_best_distance < threshold:
                    best_distance = person_best_distance
                    best_match = name

            if best_match:
                print(f"  ✓ Matched: {best_match} (distance: {best_distance:.4f})")
            else:
                print(f"  ✗ No match found (best distance: {best_distance:.4f})")

            return best_match

        except Exception as e:
            print(f"Error recognizing face: {e}")
            return None

    def recognize_face(self, face_img: np.ndarray) -> Optional[str]:
        """
        Recognize a face against known people in the database.

        Args:
            face_img: Face image as numpy array

        Returns:
            Name of recognized person or None if unknown
        """
        if self.face_app is None or len(self.db.people) == 0:
            return None

        try:
            # Get face from image
            faces = self.face_app.get(face_img)
            if not faces:
                return None

            # Use first face's embedding
            input_embedding = faces[0].embedding
            return self.recognize_face_embedding(input_embedding)

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
        if self.face_app is None:
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
            faces = self.face_app.get(face_img)
            if faces:
                return faces[0].embedding
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
            self.status = "Speaking..."
            self.speech_interface.speak(greeting)
            self.status = "Ready"

        # Ask ONE question for new person (rest will be asked in future sessions)
        for question_key in self.pending_questions:
            if question_key == "name":
                continue  # Already have the name

            # Use speech or text input
            question_text = format_question(question_key)
            if self.speech_interface:
                self.status = "Listening..."
                if self.current_frame is not None:
                    self._show_listening_indicator(self.current_frame, "LISTENING...")
                answer = self.speech_interface.ask_question(
                    question_text,
                    allow_text_fallback=True
                )
                self.status = "Ready"
            else:
                print(f"\n{question_text} ", end='', flush=True)
                answer = input().strip()

            if answer:
                info[question_key] = answer

            # Only ask ONE question
            break

        # Thank them
        thanks = "Thank you! I'll remember you."
        print(thanks)
        if self.speech_interface:
            self.status = "Speaking..."
            self.speech_interface.speak(thanks)
            self.status = "Ready"

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
            self.status = "Speaking..."
            self.speech_interface.speak(greeting)
            self.status = "Ready"

        new_info = {}

        # Ask ONE question per session (rotate through unanswered questions)
        person = self.db.get_person(name)
        for question_key in self.conversation_questions:
            # Check if we already have this info
            if person and question_key in person.metadata:
                continue  # Skip if we already know

            # Ask this ONE question
            question_text = format_question(question_key)
            if self.speech_interface:
                self.status = "Listening..."
                if self.current_frame is not None:
                    self._show_listening_indicator(self.current_frame, "LISTENING...")
                answer = self.speech_interface.ask_question(
                    question_text,
                    allow_text_fallback=True
                )
                self.status = "Ready"
            else:
                print(f"{question_text} ", end='', flush=True)
                answer = input().strip()

            if answer:
                new_info[question_key] = answer
                # Update database immediately
                self.db.update_person(name, new_info)
                print(f"✓ Learned: {question_key} = {answer}")

            # Only ask ONE question per session
            break

        if not new_info:
            print("(Already know everything about you!)")

        # Update last conversation time
        self.last_conversation[name] = current_time

        # Say goodbye
        goodbye = "Thanks for chatting!"
        if self.speech_interface:
            self.status = "Speaking..."
            self.speech_interface.speak(goodbye)
            self.status = "Ready"
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
        print("\nPress 'q' to quit, 's' to save new person, 'a' to add sample for recognized person")
        print("The agent will recognize faces and collect information.\n")

        # Track last recognized person for adding samples
        last_recognized = None
        last_face_img = None
        last_embedding = None

        # Track unknown face for auto-prompt
        unknown_face_start = None
        unknown_face_prompted = False
        auto_prompt_delay = 2.0  # seconds

        # Status display
        self.status = "Initializing..."

        self._init_camera()
        self.status = "Ready"
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

                # Store for use in other methods
                self.current_frame = frame

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

                    # Use pre-computed embedding from detect_faces
                    embedding = face_obj.get("embedding")
                    if embedding is not None:
                        recognized_name = self.recognize_face_embedding(embedding)
                    else:
                        recognized_name = self.recognize_face(face_img)

                    # Track for adding samples (store embedding for efficiency)
                    if recognized_name:
                        last_recognized = recognized_name
                        last_face_img = face_img.copy()
                        last_embedding = embedding

                    # Draw rectangle and label
                    looking = face_obj.get("looking_at_camera", False)
                    color = (0, 255, 0) if recognized_name else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)

                    # Show name and attention status
                    label = recognized_name if recognized_name else "Unknown"
                    if looking:
                        label += " [A]"  # Attention indicator
                    cv2.putText(
                        display_frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                    )

                    # Have conversation with recognized person
                    if recognized_name and self.conversation_enabled and self.conversation_questions:
                        # Trigger conversation (will check cooldown internally)
                        await self.have_conversation(recognized_name)
                        # Reset unknown tracking
                        unknown_face_start = None
                        unknown_face_prompted = False

                    # Track unknown face for auto-prompt
                    if not recognized_name and len(faces) == 1:
                        import time
                        current_time = time.time()

                        if unknown_face_start is None:
                            unknown_face_start = current_time
                            unknown_face_prompted = False

                        # Auto-prompt after 2 seconds of continuous unknown face
                        elapsed = current_time - unknown_face_start
                        auto_trigger = elapsed >= auto_prompt_delay and not unknown_face_prompted
                    else:
                        # Reset tracking if no unknown face
                        unknown_face_start = None
                        unknown_face_prompted = False
                        auto_trigger = False

                    # If unknown face detected, offer to add
                    if not recognized_name and len(faces) == 1:  # Only one face
                        cv2.imshow('Face Recognition', display_frame)
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('s') or auto_trigger:  # Manual or auto save
                            unknown_face_prompted = True
                            print("\n" + "-" * 80)
                            if auto_trigger:
                                print("Unknown face detected for 2+ seconds!")
                            else:
                                print("New person detected!")

                            # Get name via speech or text
                            if self.speech_interface:
                                self.status = "Listening for name..."
                                self._show_listening_indicator(display_frame, "LISTENING - Say your name...")
                                name = self.speech_interface.ask_question(
                                    "What is your name?",
                                    allow_text_fallback=True
                                )
                                self.status = "Ready"
                            else:
                                print("Please provide the person's name: ", end='', flush=True)
                                name = input().strip()

                            if name:
                                # Check if face image is valid
                                if face_img is None or face_img.size == 0:
                                    print("Face moved out of frame, please try again")
                                    continue

                                # Use pre-computed embedding if available
                                encoding = face_obj.get("embedding")
                                if encoding is None:
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
                                unknown_face_start = None

                        elif key == ord('q'):
                            break

                        continue

                # Draw status indicator on frame
                status_color = (0, 255, 255)  # Yellow
                cv2.putText(
                    display_frame, f"Status: {self.status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
                )

                # Display frame
                cv2.imshow('Face Recognition', display_frame)

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a') and last_recognized:
                    # Add another sample for better recognition
                    # Use pre-computed embedding if available
                    if last_embedding is not None:
                        encoding = last_embedding
                    elif last_face_img is not None:
                        encoding = self.get_face_encoding(last_face_img)
                    else:
                        encoding = None

                    if encoding is not None:
                        if self.db.add_embedding(last_recognized, encoding):
                            print(f"\n✓ Added new face sample for {last_recognized}")
                            num_samples = len(self.db.get_all_embeddings(last_recognized))
                            print(f"  Total samples: {num_samples}")
                        else:
                            print(f"\n✗ Failed to add sample for {last_recognized}")
                    else:
                        print("\n✗ Could not extract face encoding")

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
    parser.add_argument("--speech-engine", default="whisper", choices=["whisper", "google", "sphinx"],
                        help="Speech recognition engine (default: whisper)")
    parser.add_argument("--menu", action="store_true", help="Show menu instead of direct start")
    parser.add_argument("--questions", nargs="+", default=["favorite_candy", "interests", "hobby"],
                        help="Questions to ask (default: favorite_candy interests hobby)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FACE RECOGNITION AGENT (InsightFace)")
    print("=" * 80)

    # Create agent with default questions for both new people and conversations
    agent = FaceRecognitionAgent(
        speech_enabled=not args.no_speech,
        speech_engine=args.speech_engine,
        conversation_enabled=True,
        conversation_questions=args.questions
    )

    # Also set pending questions for new people
    for q in args.questions:
        agent.pending_questions.add(q)

    print(f"✓ Speech: {'Enabled' if agent.speech_interface else 'Disabled'}")
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
