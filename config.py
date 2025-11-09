"""
Configuration module for A2A agents.

Supports loading configuration from YAML, JSON, or TOML files.
Uses Pydantic for validation and type safety.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""

    name: str = Field(description="Name of the Ollama model (e.g., 'gemma3:latest')")
    base_url: str = Field(
        default="http://localhost:11434", description="Base URL for the Ollama API"
    )
    timeout: int = Field(default=60, description="Timeout in seconds for API calls")


class BiasInterrogatorConfig(BaseModel):
    """Configuration for the Bias Interrogator agent."""

    model: ModelConfig = Field(description="Model configuration for bias interrogator")
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt (optional)"
    )
    focus_areas: List[str] = Field(
        default_factory=lambda: ["gender", "race", "age", "cultural", "socioeconomic", "disability"],
        description="Bias categories to focus on",
    )


class ChatAgentConfig(BaseModel):
    """Configuration for the Chat Agent."""

    agent_id: str = Field(default="chat-agent", description="Unique identifier for the agent")
    model: ModelConfig = Field(description="Model configuration for chat agent")
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt (optional)"
    )


class SessionConfig(BaseModel):
    """Configuration for the bias testing session."""

    num_questions: int = Field(
        default=5, ge=1, le=100, description="Number of bias-probing questions to generate"
    )
    focus_area: Optional[str] = Field(
        default=None, description="Specific bias area to focus on (optional)"
    )
    mode: Literal["auto", "interactive"] = Field(
        default="auto", description="Execution mode: auto or interactive"
    )
    verbose: bool = Field(default=True, description="Enable verbose output")


class A2AConfig(BaseModel):
    """Main configuration for A2A agents."""

    bias_interrogator: BiasInterrogatorConfig = Field(
        description="Bias interrogator agent configuration"
    )
    chat_agent: ChatAgentConfig = Field(description="Chat agent configuration")
    session: SessionConfig = Field(default_factory=SessionConfig, description="Session configuration")

    @field_validator("bias_interrogator", "chat_agent", mode="before")
    @classmethod
    def validate_agents(cls, v):
        """Ensure agent configs are properly formed."""
        if isinstance(v, dict) and "model" in v:
            if isinstance(v["model"], str):
                # Allow shorthand: model: "gemma3:latest"
                v["model"] = {"name": v["model"]}
        return v

    @classmethod
    def from_file(cls, file_path: str | Path) -> "A2AConfig":
        """
        Load configuration from a file.

        Supports YAML (.yaml, .yml), JSON (.json), and TOML (.toml) formats.

        Args:
            file_path: Path to the configuration file

        Returns:
            A2AConfig instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        # Read file content
        content = file_path.read_text()
        suffix = file_path.suffix.lower()

        # Parse based on file extension
        if suffix in [".yaml", ".yml"]:
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files. Install with: pip install pyyaml"
                )
        elif suffix == ".json":
            data = json.loads(content)
        elif suffix == ".toml":
            try:
                import tomli
                data = tomli.loads(content)
            except ImportError:
                try:
                    import tomllib
                    data = tomllib.loads(content)
                except ImportError:
                    raise ImportError(
                        "tomli is required for TOML config files on Python < 3.11. "
                        "Install with: pip install tomli"
                    )
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. "
                "Supported formats: .yaml, .yml, .json, .toml"
            )

        return cls(**data)

    def to_file(self, file_path: str | Path, format: Optional[str] = None):
        """
        Save configuration to a file.

        Args:
            file_path: Path to save the configuration
            format: Output format ('yaml', 'json', or 'toml'). If None, inferred from file extension.
        """
        file_path = Path(file_path)

        if format is None:
            suffix = file_path.suffix.lower()
            if suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif suffix == ".json":
                format = "json"
            elif suffix == ".toml":
                format = "toml"
            else:
                raise ValueError(f"Cannot infer format from extension: {suffix}")

        data = self.model_dump(mode="python")

        if format == "yaml":
            try:
                import yaml
                content = yaml.dump(data, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        elif format == "json":
            content = json.dumps(data, indent=2)
        elif format == "toml":
            try:
                import tomli_w
                content = tomli_w.dumps(data)
            except ImportError:
                raise ImportError("tomli-w is required. Install with: pip install tomli-w")
        else:
            raise ValueError(f"Unsupported format: {format}")

        file_path.write_text(content)

    @classmethod
    def get_default(cls) -> "A2AConfig":
        """Get default configuration."""
        return cls(
            bias_interrogator=BiasInterrogatorConfig(
                model=ModelConfig(name="gemma3:latest")
            ),
            chat_agent=ChatAgentConfig(
                model=ModelConfig(name="gemma3:latest")
            ),
            session=SessionConfig(),
        )


def create_default_config_files():
    """Create example configuration files in different formats."""
    config = A2AConfig.get_default()

    # Create configs directory
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    # Save in different formats
    formats = {
        "config.example.yaml": "yaml",
        "config.example.json": "json",
        "config.example.toml": "toml",
    }

    for filename, fmt in formats.items():
        try:
            config.to_file(config_dir / filename, format=fmt)
            print(f"Created {config_dir / filename}")
        except ImportError as e:
            print(f"Skipping {filename}: {e}")


if __name__ == "__main__":
    # Test configuration loading
    print("Creating example configuration files...")
    create_default_config_files()
