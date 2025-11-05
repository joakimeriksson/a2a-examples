"""
Chat Agent using Google's A2A SDK.

This agent responds to general questions and can be probed for bias.
It uses the A2A SDK for agent-to-agent communication protocol.
"""

import asyncio
from typing import List, Dict, Any, Optional
import httpx
from dataclasses import dataclass, asdict
import json


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


class ChatAgent:
    """
    A general-purpose chat agent using Google's A2A SDK conventions.

    This agent responds to questions on any topic and follows the
    Agent-to-Agent (A2A) communication protocol.
    """

    def __init__(
        self,
        agent_id: str = "chat-agent",
        model_name: str = "gemma3:latest",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the Chat Agent.

        Args:
            agent_id: Unique identifier for this agent
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history: List[A2AMessage] = []
        self.system_prompt = """You are a helpful, knowledgeable assistant that answers
        questions on any topic. Be informative, accurate, and balanced in your responses.
        Avoid making assumptions based on stereotypes or biases."""

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> A2AResponse:
        """
        Process an incoming message using A2A protocol.

        Args:
            message: The incoming message to process
            context: Optional context information

        Returns:
            A2AResponse with the agent's reply
        """
        # Add message to conversation history
        user_message = A2AMessage(role="user", content=message, metadata=context)
        self.conversation_history.append(user_message)

        # Generate response using Ollama
        response_text = await self._generate_response(message)

        # Create A2A response
        assistant_message = A2AMessage(role="assistant", content=response_text)
        self.conversation_history.append(assistant_message)

        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.utcnow().isoformat()

        return A2AResponse(
            message=response_text,
            agent_id=self.agent_id,
            timestamp=timestamp,
            metadata={"model": self.model_name, "context": context},
        )

    async def _generate_response(self, message: str) -> str:
        """
        Generate a response using Ollama.

        Args:
            message: The message to respond to

        Returns:
            Generated response text
        """
        # Build messages for Ollama API
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        for msg in self.conversation_history[-10:]:  # Keep last 10 messages
            messages.append({"role": msg.role, "content": msg.content})

        # Add current message if not already in history
        if not self.conversation_history or self.conversation_history[-1].content != message:
            messages.append({"role": "user", "content": message})

        # Call Ollama API
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
                return f"Error generating response: {str(e)}"

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history in serializable format.

        Returns:
            List of conversation messages
        """
        return [asdict(msg) for msg in self.conversation_history]

    async def handle_a2a_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an A2A protocol request.

        Args:
            request_data: The A2A request data

        Returns:
            A2A response data
        """
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
    print("Initializing Chat Agent (Google A2A SDK)...")
    agent = ChatAgent()

    # Test questions
    test_questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What are the benefits of exercise?",
    ]

    print("\n" + "=" * 80)
    print("Testing Chat Agent with Sample Questions:")
    print("=" * 80)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = await agent.process_message(question)
        print(f"Response: {response.message}")
        print(f"Agent ID: {response.agent_id}")
        print(f"Timestamp: {response.timestamp}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
