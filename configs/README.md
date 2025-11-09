# Configuration Files

This directory contains example configuration files for the A2A agents.

## Quick Start

1. Copy an example config file:
```bash
cp configs/config.example.yaml config.yaml
```

2. Edit the configuration to your needs

3. Run with the config file:
```bash
python main.py --config config.yaml
# or with pixi
pixi run run -- --config config.yaml
```

## Configuration Formats

You can use any of these formats:
- **YAML** (`.yaml`, `.yml`) - Recommended, most readable
- **JSON** (`.json`) - Good for programmatic generation
- **TOML** (`.toml`) - Alternative, similar to YAML

## Example Files

### `config.example.yaml` / `config.example.json` / `config.example.toml`
Basic configuration using the same model (gemma3:latest) for both agents.

### `multi-model.example.yaml`
Advanced example showing:
- Different models for each agent (Llama for interrogator, Gemma for chat)
- Custom system prompts
- Focused bias testing (gender only)
- Longer timeouts for larger models

## Configuration Options

### Bias Interrogator

```yaml
bias_interrogator:
  model:
    name: gemma3:latest           # Ollama model name
    base_url: http://localhost:11434  # Ollama API URL
    timeout: 60                   # Timeout in seconds
  system_prompt: "Custom prompt..."  # Optional custom system prompt
  focus_areas:                    # Bias categories to test
    - gender
    - race
    - age
    - cultural
    - socioeconomic
    - disability
```

### Chat Agent

```yaml
chat_agent:
  agent_id: chat-agent            # Unique agent identifier
  model:
    name: gemma3:latest
    base_url: http://localhost:11434
    timeout: 60
  system_prompt: "Custom prompt..."  # Optional custom system prompt
```

### Session

```yaml
session:
  num_questions: 5                # Number of questions to generate (1-100)
  focus_area: gender              # Optional: focus on specific bias area
  mode: auto                      # 'auto' or 'interactive'
  verbose: true                   # Enable verbose output
```

## Shorthand Syntax

You can use shorthand for model configuration:

```yaml
bias_interrogator:
  model: gemma3:latest  # Shorthand - uses default base_url and timeout
```

This is equivalent to:

```yaml
bias_interrogator:
  model:
    name: gemma3:latest
    base_url: http://localhost:11434
    timeout: 60
```

## Model Selection Tips

### For Bias Interrogator
- **llama3.2** or **llama3.1** - Best for nuanced question generation
- **gemma3:latest** - Good balance of speed and quality
- **phi3** - Fast, lighter weight option

### For Chat Agent
- **gemma3:latest** - Good general purpose responses
- **llama3.2** - More detailed, thoughtful responses
- **mistral** - Good reasoning capabilities

### Testing Different Models
Create multiple config files to test different combinations:

```bash
# Test with same model
python main.py --config configs/config.example.yaml

# Test with different models
python main.py --config configs/multi-model.example.yaml
```

## Remote Ollama

If running Ollama on a different machine:

```yaml
bias_interrogator:
  model:
    name: gemma3:latest
    base_url: http://192.168.1.100:11434  # Remote Ollama server
```

## Environment-Specific Configs

Create different configs for different scenarios:

```
configs/
├── dev.yaml          # Development settings
├── production.yaml   # Production settings
├── testing.yaml      # Testing with smaller models
└── research.yaml     # Research with larger models
```

Then use them:
```bash
python main.py --config configs/dev.yaml
python main.py --config configs/production.yaml
```
