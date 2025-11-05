"""
Quick test script to verify both agents are working.

This script performs a simple test of both agents without running
the full bias interrogation workflow.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.bias_interrogator import BiasInterrogator
from agents.chat_agent import ChatAgent


async def test_bias_interrogator():
    """Test the bias interrogator agent."""
    print("\n" + "=" * 80)
    print("Testing Bias Interrogator Agent (pydantic-ai)")
    print("=" * 80)

    try:
        interrogator = BiasInterrogator()
        print("✓ Bias Interrogator initialized successfully")

        print("\nGenerating 2 test questions...")
        questions = await interrogator.generate_questions(num_questions=2)

        print(f"✓ Generated {len(questions.questions)} questions\n")

        for i, q in enumerate(questions.questions, 1):
            print(f"{i}. {q.question}")
            print(f"   Category: {q.category}")
            print(f"   Rationale: {q.rationale}\n")

        return True

    except Exception as e:
        print(f"✗ Error testing Bias Interrogator: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_chat_agent():
    """Test the chat agent."""
    print("\n" + "=" * 80)
    print("Testing Chat Agent (A2A SDK)")
    print("=" * 80)

    try:
        agent = ChatAgent()
        print("✓ Chat Agent initialized successfully")

        test_question = "What is 2+2?"
        print(f"\nAsking: {test_question}")

        response = await agent.process_message(test_question)

        print(f"✓ Response received: {response.message[:100]}...")
        print(f"✓ Agent ID: {response.agent_id}")
        print(f"✓ Timestamp: {response.timestamp}")

        return True

    except Exception as e:
        print(f"✗ Error testing Chat Agent: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("QUICK TEST: A2A AGENTS")
    print("=" * 80)
    print("\nThis script tests both agents to ensure they're working correctly.")
    print("Make sure Ollama is running with the gemma2:2b model installed.\n")

    # Test both agents
    results = []
    results.append(await test_bias_interrogator())
    results.append(await test_chat_agent())

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if all(results):
        print("✓ All tests passed!")
        print("\nYou can now run the full example with: python main.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nMake sure:")
        print("  1. Ollama is running (ollama serve)")
        print("  2. gemma2:2b model is installed (ollama pull gemma2:2b)")
        print("  3. All dependencies are installed (pip install -r requirements.txt)")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
