# Face Recognition Agent

A webcam-based face recognition agent that can identify people, store their information, and respond to queries from other agents using the A2A (Agent-to-Agent) protocol.

## Features

- **Real-time Face Recognition**: Uses webcam to detect and recognize people
- **Deep Learning Model**: Powered by DeepFace with multiple backend options (Facenet512, VGG-Face, ArcFace, etc.)
- **Speech Recognition**: Collect information via voice input using Google Speech Recognition, Whisper, or Sphinx
- **Text-to-Speech**: Audio prompts for hands-free interaction
- **Persistent Storage**: Stores face encodings, photos, and metadata on disk
- **Interactive Information Collection**: Asks questions to gather information about people (via speech or text)
- **A2A Protocol Support**: Other agents can query person information and request additional questions
- **Dynamic Question Collection**: Agents can request specific information to be collected when people are recognized

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Face Recognition Agent (A2A)                       │
│                                                                 │
│  ┌──────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Webcam     │───►│  DeepFace   │───►│   Recognition   │  │
│  │   Capture    │    │  Detection  │    │     Engine      │  │
│  └──────────────┘    └─────────────┘    └─────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Person Database                              │ │
│  │  • Face Encodings                                        │ │
│  │  • Photos                                                │ │
│  │  • Metadata (name, interests, preferences, etc.)         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              A2A Protocol Interface                       │ │
│  │  • query_person                                          │ │
│  │  • list_people                                           │ │
│  │  • request_questions                                     │ │
│  │  • search_people                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Other A2A      │
                    │  Agents         │
                    │                 │
                    │ • Food Prefs    │
                    │ • Music Tastes  │
                    │ • Chat Agents   │
                    └─────────────────┘
```

## Prerequisites

1. **Python 3.10+**
2. **Webcam** - Required for face recognition
3. **Microphone** - Required for speech recognition (optional feature)
4. **Ollama** (Optional) - For natural language processing tasks
5. **GPU** (Optional but recommended) - For faster face recognition

## Installation

### Using Pixi (Recommended)

```bash
cd face_recognition
pixi install
```

### Using pip

Install the face recognition specific dependencies:

```bash
cd face_recognition
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Face Recognition

Run the agent in interactive mode:

```bash
cd face_recognition

# Using pixi
pixi run run

# Or using python directly
python face_recognition_agent.py
```

This will:
1. Open your webcam
2. Detect faces in real-time
3. Recognize known faces
4. Allow you to add new people by pressing 's'
5. Store photos and information

### 2. Example Usage Script

Run the comprehensive example:

```bash
cd face_recognition

# Using pixi
pixi run examples

# Or using python directly
python example_usage.py
```

This provides several demos:
- Basic face recognition
- A2A protocol queries
- Agent-to-agent interaction simulation

## Usage Guide

### Adding New People

1. Start the face recognition agent
2. Look at the webcam
3. When a face is detected (red box), press 's' to save
4. **Speak** or type the person's name when prompted (speech recognition enabled by default)
5. Answer any additional questions (speak or type - text fallback always available)
6. The person is now stored in the database

### Recognizing People

- Known faces are shown with a green box and their name
- Unknown faces are shown with a red box and "Unknown" label
- Press 'q' to quit the recognition loop

### Having Conversations

When a person is recognized, the agent can automatically start a conversation:

1. **Configure conversation questions**:
   ```python
   agent = FaceRecognitionAgent(
       conversation_enabled=True,
       conversation_questions=["favorite_candy", "interests", "hobby"]
   )
   ```

2. **The agent will**:
   - Greet the person by name: "Hello John! Nice to see you again!"
   - Ask configured questions they haven't answered yet
   - Update their profile with new information
   - Wait 5 minutes (configurable) before asking again

3. **Example**:
   ```
   ============================================================
   Hello John! Nice to see you again!
   🔊 (Speaking: "Tell me, what is your favorite candy?")
   🎤 Listening... (speak now)
   ✓ Heard: Chocolate
   Thanks for chatting!
   ============================================================
   ```

### A2A Protocol Operations

Other agents can interact with the Face Recognition Agent using these operations:

#### 1. Query Person

Get detailed information about a specific person:

```python
request = {
    "operation": "query_person",
    "name": "John Doe"
}
response = await agent.handle_a2a_request(request)
```

Response:
```json
{
    "status": "success",
    "person": {
        "name": "John Doe",
        "metadata": {
            "favorite_food": "Pizza",
            "hobby": "Photography"
        },
        "first_seen": "2025-01-01T10:30:00",
        "last_seen": "2025-01-05T14:20:00"
    }
}
```

#### 2. List All People

Get a list of all known people:

```python
request = {
    "operation": "list_people"
}
response = await agent.handle_a2a_request(request)
```

Response:
```json
{
    "status": "success",
    "people": ["John Doe", "Jane Smith", "Bob Wilson"],
    "count": 3
}
```

#### 3. Request Questions

Ask the agent to collect specific information when meeting people:

```python
request = {
    "operation": "request_questions",
    "questions": ["favorite_food", "favorite_color", "hobby"]
}
response = await agent.handle_a2a_request(request)
```

Response:
```json
{
    "status": "success",
    "message": "Added 3 questions to collection queue",
    "pending_questions": ["name", "favorite_food", "favorite_color", "hobby"]
}
```

The agent will now ask these questions when adding new people or updating existing ones.

#### 4. Search People

Search for people based on metadata:

```python
request = {
    "operation": "search_people",
    "query": {
        "favorite_food": "Pizza"
    }
}
response = await agent.handle_a2a_request(request)
```

Response:
```json
{
    "status": "success",
    "results": [
        {
            "name": "John Doe",
            "metadata": {"favorite_food": "Pizza", "hobby": "Photography"}
        }
    ],
    "count": 1
}
```

#### 5. Set Conversation Questions

Configure which questions to ask recognized people:

```python
request = {
    "operation": "set_conversation_questions",
    "questions": ["favorite_candy", "interests", "hobby"],
    "replace": False  # True to replace all, False to add
}
response = await agent.handle_a2a_request(request)
```

Response:
```json
{
    "status": "success",
    "message": "Conversation questions updated to include 3 questions",
    "conversation_questions": ["favorite_candy", "interests", "hobby"]
}
```

#### 6. Get Conversation Configuration

Get current conversation settings:

```python
request = {
    "operation": "get_conversation_config"
}
response = await agent.handle_a2a_request(request)
```

Response:
```json
{
    "status": "success",
    "conversation_enabled": true,
    "conversation_questions": ["favorite_candy", "interests"],
    "conversation_cooldown": 300
}
```

#### 7. Set Conversation Configuration

Change conversation settings:

```python
request = {
    "operation": "set_conversation_config",
    "enabled": true,
    "cooldown": 60  # seconds between conversations
}
response = await agent.handle_a2a_request(request)
```

## Configuration

### Speech Recognition

Speech recognition is **enabled by default** and supports multiple engines:

```python
agent = FaceRecognitionAgent(
    speech_enabled=True,          # Enable/disable speech recognition
    speech_engine="google",       # Options: google, whisper, sphinx
    tts_enabled=True              # Enable/disable text-to-speech prompts
)
```

**Available Speech Engines:**

1. **Google Speech Recognition** (default)
   - Online service (requires internet)
   - Fast and accurate
   - Free for moderate use
   - Best for most users

2. **Whisper** (OpenAI)
   - Offline (after model download)
   - Excellent accuracy
   - Supports multiple languages
   - Requires more resources
   - First run downloads model (~140MB for base model)

3. **Sphinx** (CMU)
   - Fully offline
   - Lower accuracy
   - Lightweight
   - Good for privacy-sensitive applications

**Disabling Speech Recognition:**

```python
# Use text input only
agent = FaceRecognitionAgent(speech_enabled=False)
```

**Note:** Even with speech recognition enabled, you can always type instead of speaking. The agent will automatically fall back to text input if speech recognition fails.

### Conversation Configuration

Configure automatic conversations with recognized people:

```python
agent = FaceRecognitionAgent(
    conversation_enabled=True,                    # Enable conversations
    conversation_questions=["favorite_candy", "interests"],  # Questions to ask
)
```

**Configuration Options:**

- `conversation_enabled`: Enable/disable conversations (default: `True`)
- `conversation_questions`: List of questions to ask (default: `[]`)
- `conversation_cooldown`: Seconds between conversations with same person (default: `300`)

**Example Questions:**
- `"favorite_candy"` - What candy do you like?
- `"interests"` - What are your interests?
- `"hobby"` - What hobbies do you have?
- `"favorite_music"` - What music do you like?
- `"sport"` - Do you play any sports?

**Dynamic Configuration via A2A:**

Other agents can configure conversations at runtime:

```python
# Add questions
await agent.handle_a2a_request({
    "operation": "set_conversation_questions",
    "questions": ["favorite_candy"],
    "replace": False  # Add to existing
})

# Change cooldown
await agent.handle_a2a_request({
    "operation": "set_conversation_config",
    "cooldown": 60  # 1 minute
})
```

### Face Recognition Models

You can configure the face recognition model when creating the agent:

```python
agent = FaceRecognitionAgent(
    face_model="Facenet512",  # Options: Facenet512, VGG-Face, ArcFace, Dlib, etc.
    detector_backend="opencv"  # Options: opencv, ssd, dlib, mtcnn, retinaface
)
```

**Available Models:**
- `Facenet512` (default) - Fast and accurate
- `VGG-Face` - High accuracy, slower
- `ArcFace` - State-of-the-art accuracy
- `Dlib` - Traditional approach
- `OpenFace` - Lightweight
- `Facenet` - Good balance

**Available Detectors:**
- `opencv` (default) - Fast, good for simple scenarios
- `ssd` - Better accuracy, slower
- `mtcnn` - Multi-task CNN, very accurate
- `retinaface` - Best accuracy, slower
- `dlib` - Traditional approach

### Data Storage

By default, data is stored in `people_data/`:

```
people_data/
├── database.json          # Person records and metadata
├── photos/                # Stored face photos
│   └── John_Doe_2025-01-01T10-30-00.jpg
└── encodings/             # Face encodings (numpy arrays)
    └── John_Doe_2025-01-01T10-30-00.npy
```

You can configure the storage location:

```python
agent = FaceRecognitionAgent(data_dir="custom_data_directory")
```

## Example: Agent-to-Agent Interaction

Here's a complete example of how another agent can interact with the Face Recognition Agent:

```python
import asyncio
from face_recognition.face_recognition_agent import FaceRecognitionAgent

async def food_preference_agent():
    """Example agent that queries food preferences."""

    # Create face recognition agent instance
    face_agent = FaceRecognitionAgent()

    # Request that food preferences be collected
    await face_agent.handle_a2a_request({
        "operation": "request_questions",
        "questions": ["favorite_food", "dietary_restrictions", "allergies"]
    })

    # Later, query for people who like pizza
    result = await face_agent.handle_a2a_request({
        "operation": "search_people",
        "query": {"favorite_food": "Pizza"}
    })

    print(f"People who like pizza: {result['results']}")

    # Get detailed info about a specific person
    if result['results']:
        person_name = result['results'][0]['name']
        person_info = await face_agent.handle_a2a_request({
            "operation": "query_person",
            "name": person_name
        })
        print(f"Details: {person_info}")

asyncio.run(food_preference_agent())
```

## Use Cases

### 1. Smart Home Assistant
- Recognize family members
- Adjust room settings based on preferences
- Track presence for security

### 2. Customer Service
- Recognize returning customers
- Remember previous interactions
- Personalize service

### 3. Healthcare
- Patient identification
- Track patient visits
- Remember patient preferences

### 4. Event Management
- Check-in attendees
- Track networking connections
- Remember dietary restrictions

### 5. Personal Assistant
- Remember names and faces
- Track interests and preferences
- Suggest relevant information

## Troubleshooting

### Speech Recognition Not Working

**PyAudio Installation Issues:**

PyAudio is required for microphone access. Installation varies by platform:

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
pip install pipwin
pipwin install pyaudio
```

**Microphone Permissions:**

- **macOS**: Grant Terminal/IDE microphone access in System Preferences → Security & Privacy → Microphone
- **Linux**: Ensure your user is in the `audio` group: `sudo usermod -a -G audio $USER`
- **Windows**: Check microphone privacy settings

**Test Microphone:**

```bash
# Test speech recognition
python speech_interface.py
```

**Whisper Model Download:**

First use of Whisper will download models:
```bash
# Downloads happen automatically, but you can pre-download:
python -c "import whisper; whisper.load_model('base')"
```

### Camera Not Opening

```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Failed')"
```

### DeepFace Installation Issues

If you encounter TensorFlow/Keras issues:

```bash
pip install --upgrade tensorflow
pip install tf-keras
```

For macOS with M1/M2:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

### Face Not Recognized After Saving

If you save a person but they're not recognized:

1. **Check the debug output**: The agent prints distance values when comparing faces
   - Look for lines like: `Comparing with Name: distance=0.1234, threshold=0.30`
   - If distance is close to threshold, try adjusting the threshold

2. **Adjust recognition threshold**: Edit `face_recognition_agent.py` line ~372
   ```python
   # Make recognition more lenient (increase threshold)
   "Facenet512": 0.40,  # Default is 0.30, try 0.35-0.45
   ```

3. **Ensure good conditions when saving AND recognizing**:
   - Good, consistent lighting
   - Face clearly visible (not blurry)
   - Look directly at camera
   - Avoid extreme angles

4. **Try a different face model**: Some models are more robust
   ```python
   agent = FaceRecognitionAgent(face_model="ArcFace")  # More accurate but slower
   ```

5. **Check saved data**: Verify encodings were saved
   ```bash
   ls -la people_data/encodings/
   ls -la people_data/photos/
   ```

### Low Recognition Accuracy

1. Ensure good lighting conditions
2. Try a different face model (e.g., ArcFace for better accuracy)
3. Try a different detector (e.g., mtcnn or retinaface)
4. Ensure the camera is focused on the face
5. Check that face images are at least 50x50 pixels

### Performance Issues

1. Use a GPU if available
2. Switch to a faster model (e.g., OpenFace)
3. Use opencv detector for faster detection
4. Lower the camera resolution
5. For speech: Use Google engine instead of Whisper for faster response

## API Reference

### FaceRecognitionAgent

```python
class FaceRecognitionAgent:
    def __init__(
        self,
        agent_id: str = "face-recognition-agent",
        model_name: str = "gemma3:latest",
        base_url: str = "http://localhost:11434",
        data_dir: str = "people_data",
        face_model: str = "Facenet512",
        detector_backend: str = "opencv"
    )
```

**Methods:**

- `capture_frame() -> Optional[np.ndarray]` - Capture a frame from webcam
- `detect_faces(frame: np.ndarray) -> List[Dict]` - Detect faces in a frame
- `recognize_face(face_img: np.ndarray) -> Optional[str]` - Recognize a face
- `process_recognition_loop(duration: int = 30)` - Run recognition loop
- `handle_a2a_request(request_data: Dict) -> Dict` - Handle A2A requests

### PersonDatabase

```python
class PersonDatabase:
    def __init__(self, data_dir: str = "people_data")
```

**Methods:**

- `add_person(name, photo, encoding, metadata) -> PersonInfo` - Add new person
- `update_person(name, metadata)` - Update person metadata
- `get_person(name) -> Optional[PersonInfo]` - Get person by name
- `list_people() -> List[str]` - List all known people
- `search_by_metadata(query: Dict) -> List[PersonInfo]` - Search by metadata

## Privacy and Security

**Important Considerations:**

1. **Data Storage**: Face encodings and photos are stored locally. Ensure proper file permissions.
2. **Consent**: Always obtain consent before capturing and storing someone's face.
3. **Data Protection**: Consider encrypting stored data for sensitive applications.
4. **Access Control**: Implement authentication for A2A requests in production.
5. **Retention Policy**: Implement data retention and deletion policies.

## Performance Benchmarks

Typical performance on consumer hardware:

| Model | Detector | Speed (fps) | Accuracy |
|-------|----------|-------------|----------|
| Facenet512 | opencv | 20-30 | Good |
| Facenet512 | mtcnn | 10-15 | Excellent |
| ArcFace | retinaface | 5-10 | Best |
| OpenFace | opencv | 30-40 | Fair |
| VGG-Face | ssd | 8-12 | Very Good |

*Benchmarks on: Intel i7, 16GB RAM, no GPU*

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add REST API server
- [ ] Implement face clustering
- [ ] Add emotion recognition
- [ ] Support multiple cameras
- [ ] Add face tracking across frames
- [ ] Implement privacy-preserving features
- [ ] Add web interface

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) - Deep learning face recognition library
- [OpenCV](https://opencv.org/) - Computer vision library
- [TensorFlow](https://www.tensorflow.org/) - Machine learning framework
- A2A Protocol - Agent-to-Agent communication standard

## Further Reading

- [DeepFace Documentation](https://github.com/serengil/deepface)
- [Face Recognition Best Practices](https://www.nist.gov/programs-projects/face-recognition-vendor-test-frvt)
- [Privacy in Face Recognition](https://www.eff.org/issues/face-recognition)
- [A2A Protocol Documentation](https://developers.google.com/agent-protocol)
