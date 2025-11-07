# A2A Examples

A collection of Agent-to-Agent (A2A) communication examples demonstrating different frameworks and use cases.

## Examples

### [Bias Agents](./bias-agents/)

Demonstrates A2A interaction between two agents:
- **Bias Interrogator** (built with pydantic-ai) - Generates questions to detect bias
- **Chat Agent** (built with Google A2A SDK) - Responds to questions

Both agents communicate using local Ollama models to create an automated bias testing system.

[View detailed documentation →](./bias-agents/README.md)

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **gemma3:latest model**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull gemma3:latest
```

### Installation

#### Using Pixi (Recommended)

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install
```

#### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r bias-agents/requirements.txt
```

## Running Examples

### Bias Agents Example

```bash
# Using pixi
pixi run run

# Or using python directly
python bias-agents/main.py
```

For more options and detailed usage, see the [bias-agents README](./bias-agents/README.md).

## Project Structure

```
a2a-examples/
├── bias-agents/           # Bias detection agents example
│   ├── agents/           # Agent implementations
│   ├── examples/         # Example scripts
│   ├── main.py          # Main orchestrator
│   └── README.md        # Detailed documentation
├── pixi.toml            # Pixi configuration
└── README.md            # This file
```

## Contributing

Contributions are welcome! Feel free to add new A2A examples or improve existing ones.

## License

MIT License - See LICENSE file for details
