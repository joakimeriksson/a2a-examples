# A2A Examples

A collection of Agent-to-Agent (A2A) communication examples demonstrating different frameworks and use cases.

Each example is self-contained with its own dependencies, configuration, and documentation.

## Examples

### [Bias Agents](./bias-agents/)

Demonstrates A2A interaction between two agents:
- **Bias Interrogator** (built with pydantic-ai) - Generates questions to detect bias
- **Chat Agent** (built with Google A2A SDK) - Responds to questions

Both agents communicate using local Ollama models to create an automated bias testing system.

[View detailed documentation →](./bias-agents/README.md)

## Quick Start

Each example is self-contained and can be run independently. Navigate to the example folder and follow its README.

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **Model for the example** (e.g., gemma3:latest)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull gemma3:latest
```

### Running the Bias Agents Example

#### Using Pixi (Recommended)

```bash
cd bias-agents
pixi install
pixi run run
```

#### Using pip

```bash
cd bias-agents
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

For more options and detailed usage, see the [bias-agents README](./bias-agents/README.md).

## Project Structure

```
a2a-examples/
├── bias-agents/           # Bias detection agents example
│   ├── agents/           # Agent implementations
│   ├── examples/         # Example scripts
│   ├── main.py          # Main orchestrator
│   ├── pixi.toml        # Pixi configuration
│   ├── requirements.txt  # Python dependencies
│   └── README.md        # Detailed documentation
├── LICENSE               # MIT License
└── README.md            # This file
```

## Contributing

Contributions are welcome! Feel free to add new A2A examples or improve existing ones.

When adding a new example:
1. Create a new folder at the root level
2. Include a complete `pixi.toml`, `requirements.txt`, and `README.md`
3. Make the example self-contained with all necessary dependencies
4. Update this README to list your example

## License

MIT License - See LICENSE file for details
