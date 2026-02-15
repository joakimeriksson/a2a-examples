"""
Chat Agent supporting multiple LLM backends.

Supports:
- Ollama (local, default)
- OpenAI-compatible APIs (Together AI, Fireworks, DeepSeek, DashScope/Qwen, etc.)

This agent responds to questions and can be probed for bias evaluation.
"""

import asyncio
from typing import List, Dict, Any, Optional
import httpx
from dataclasses import dataclass, asdict
from datetime import datetime


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


# Well-known OpenAI-compatible API base URLs
KNOWN_PROVIDERS = {
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class ChatAgent:
    """
    A general-purpose chat agent supporting Ollama and OpenAI-compatible APIs.
    """

    def __init__(
        self,
        agent_id: str = "chat-agent",
        model_name: str = "gemma3:latest",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        backend: str = "ollama",
    ):
        """
        Initialize the Chat Agent.

        Args:
            agent_id: Unique identifier for this agent
            model_name: Model name/ID (e.g., "gemma3:latest", "meta-llama/Llama-3-70b", "deepseek-chat")
            base_url: API base URL. For Ollama: http://localhost:11434.
                      For OpenAI-compatible: the provider's base URL or a key from KNOWN_PROVIDERS.
            api_key: API key for authenticated endpoints (not needed for Ollama)
            backend: "ollama" or "openai" (openai = OpenAI-compatible API)
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.api_key = api_key
        self.backend = backend
        self.conversation_history: List[A2AMessage] = []
        self.system_prompt = """You are a helpful, knowledgeable assistant that answers
        questions on any topic. Be informative and accurate in your responses."""

        # Resolve provider shorthand to URL
        if base_url in KNOWN_PROVIDERS:
            self.base_url = KNOWN_PROVIDERS[base_url]
            self.backend = "openai"
        else:
            self.base_url = base_url

        # Auto-detect backend if api_key is provided
        if api_key and backend == "ollama":
            self.backend = "openai"

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> A2AResponse:
        """Process an incoming message and return a response."""
        user_message = A2AMessage(role="user", content=message, metadata=context)
        self.conversation_history.append(user_message)

        response_text = await self._generate_response(message)

        assistant_message = A2AMessage(role="assistant", content=response_text)
        self.conversation_history.append(assistant_message)

        timestamp = datetime.utcnow().isoformat()

        return A2AResponse(
            message=response_text,
            agent_id=self.agent_id,
            timestamp=timestamp,
            metadata={"model": self.model_name, "backend": self.backend, "context": context},
        )

    def _build_messages(self, message: str) -> List[Dict[str, str]]:
        """Build the messages list for the API call."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in self.conversation_history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})

        if not self.conversation_history or self.conversation_history[-1].content != message:
            messages.append({"role": "user", "content": message})

        return messages

    async def _generate_response(self, message: str) -> str:
        """Generate a response using the configured backend."""
        if self.backend == "openai":
            return await self._generate_openai_compatible(message)
        else:
            return await self._generate_ollama(message)

    async def _generate_ollama(self, message: str) -> str:
        """Generate via Ollama API."""
        messages = self._build_messages(message)

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.model_name, "messages": messages, "stream": False},
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "")
            except Exception as e:
                return f"Error generating response: {str(e)}"

    async def _generate_openai_compatible(self, message: str) -> str:
        """Generate via OpenAI-compatible API (Together, Fireworks, DeepSeek, etc.)."""
        messages = self._build_messages(message)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": 0.7,
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                return f"API error ({e.response.status_code}): {e.response.text[:200]}"
            except Exception as e:
                return f"Error generating response: {str(e)}"

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history in serializable format."""
        return [asdict(msg) for msg in self.conversation_history]

    async def handle_a2a_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an A2A protocol request."""
        message = request_data.get("message", "")
        context = request_data.get("context", {})

        response = await self.process_message(message, context)

        return {
            "status": "success",
            "response": asdict(response),
            "agent_info": {
                "id": self.agent_id,
                "type": "chat_agent",
                "capabilities": ["general_qa", "a2a_communication"],
            },
        }


async def main():
    """Example usage of the Chat Agent."""
    print("Initializing Chat Agent...")
    agent = ChatAgent()

    test_questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]

    print("\n" + "=" * 80)
    print("Testing Chat Agent:")
    print("=" * 80)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = await agent.process_message(question)
        print(f"Response: {response.message[:200]}...")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
