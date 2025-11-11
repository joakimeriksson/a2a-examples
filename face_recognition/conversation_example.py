"""
Example showing how to use conversation features with the Face Recognition Agent.

This demonstrates:
1. Setting up conversation questions
2. Configuring via A2A protocol
3. Having conversations with recognized people
"""

import asyncio
import json
from face_recognition_agent import FaceRecognitionAgent


async def example_basic_conversation():
    """Example: Basic conversation setup."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Conversation Setup")
    print("=" * 80)

    # Create agent with conversation questions
    agent = FaceRecognitionAgent(
        conversation_enabled=True,
        conversation_questions=["favorite_candy", "interests", "hobby"]
    )

    print("\n✓ Agent created with conversation questions:")
    for q in agent.conversation_questions:
        print(f"  - {q}")

    print("\nWhen a person is recognized, the agent will:")
    print("  1. Greet them by name")
    print("  2. Ask any questions they haven't answered yet")
    print("  3. Update their profile")
    print("  4. Remember not to ask again for 5 minutes")

    return agent


async def example_a2a_configuration():
    """Example: Configure conversation via A2A protocol."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Configure via A2A Protocol")
    print("=" * 80)

    # Create basic agent
    agent = FaceRecognitionAgent()

    # Simulate another agent requesting conversation questions
    print("\n[Candy Agent] Requesting favorite candy information...")

    request = {
        "operation": "set_conversation_questions",
        "questions": ["favorite_candy", "least_favorite_candy"]
    }

    response = await agent.handle_a2a_request(request)
    print(f"\n[Face Recognition Agent] Response:")
    print(json.dumps(response, indent=2))

    # Get current configuration
    print("\n[System] Checking conversation configuration...")

    request = {
        "operation": "get_conversation_config"
    }

    response = await agent.handle_a2a_request(request)
    print(f"\n[Face Recognition Agent] Current config:")
    print(json.dumps(response, indent=2))

    return agent


async def example_multiple_agents():
    """Example: Multiple agents requesting different information."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multiple Agents Requesting Information")
    print("=" * 80)

    agent = FaceRecognitionAgent()

    # Candy preference agent
    print("\n[Candy Agent] I want to know about candy preferences...")
    response = await agent.handle_a2a_request({
        "operation": "set_conversation_questions",
        "questions": ["favorite_candy", "favorite_chocolate"]
    })
    print(f"  → {response['message']}")

    # Music preference agent
    print("\n[Music Agent] I want to know about music taste...")
    response = await agent.handle_a2a_request({
        "operation": "set_conversation_questions",
        "questions": ["favorite_music", "favorite_artist"]
    })
    print(f"  → {response['message']}")

    # Hobby agent
    print("\n[Hobby Agent] I want to know about hobbies...")
    response = await agent.handle_a2a_request({
        "operation": "set_conversation_questions",
        "questions": ["hobby", "sport"]
    })
    print(f"  → {response['message']}")

    # Check final configuration
    response = await agent.handle_a2a_request({
        "operation": "get_conversation_config"
    })

    print(f"\n[System] Final conversation questions:")
    for q in response['conversation_questions']:
        print(f"  - {q}")

    print("\nNow when the agent recognizes someone, it will ask ALL these questions!")

    return agent


async def example_replace_questions():
    """Example: Replacing vs. adding questions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Replace vs. Add Questions")
    print("=" * 80)

    agent = FaceRecognitionAgent(
        conversation_questions=["favorite_color", "favorite_food"]
    )

    print(f"\nInitial questions: {agent.conversation_questions}")

    # Add new questions
    print("\n1. Adding questions (keeps existing)...")
    response = await agent.handle_a2a_request({
        "operation": "set_conversation_questions",
        "questions": ["favorite_candy"],
        "replace": False  # Add to existing
    })
    print(f"  → {response['conversation_questions']}")

    # Replace all questions
    print("\n2. Replacing questions (removes existing)...")
    response = await agent.handle_a2a_request({
        "operation": "set_conversation_questions",
        "questions": ["favorite_movie", "favorite_book"],
        "replace": True  # Replace all
    })
    print(f"  → {response['conversation_questions']}")

    return agent


async def example_configuration_changes():
    """Example: Changing conversation settings."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Dynamic Configuration")
    print("=" * 80)

    agent = FaceRecognitionAgent(
        conversation_enabled=True,
        conversation_questions=["favorite_candy"]
    )

    print(f"\nInitial cooldown: {agent.conversation_cooldown} seconds")

    # Change cooldown to 1 minute
    print("\nChanging cooldown to 60 seconds...")
    response = await agent.handle_a2a_request({
        "operation": "set_conversation_config",
        "cooldown": 60
    })
    print(f"  → New cooldown: {response['conversation_cooldown']} seconds")

    # Disable conversations
    print("\nDisabling conversations...")
    response = await agent.handle_a2a_request({
        "operation": "set_conversation_config",
        "enabled": False
    })
    print(f"  → Conversations enabled: {response['conversation_enabled']}")

    # Re-enable conversations
    print("\nRe-enabling conversations...")
    response = await agent.handle_a2a_request({
        "operation": "set_conversation_config",
        "enabled": True
    })
    print(f"  → Conversations enabled: {response['conversation_enabled']}")

    return agent


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FACE RECOGNITION AGENT - CONVERSATION EXAMPLES")
    print("=" * 80)
    print("\nThese examples show how to configure and use conversation features.")
    print("The agent can ask questions when it recognizes people!")

    # Run examples
    await example_basic_conversation()
    await example_a2a_configuration()
    await example_multiple_agents()
    await example_replace_questions()
    await example_configuration_changes()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nTo test with real face recognition:")
    print("  1. Configure conversation questions (as shown above)")
    print("  2. Run: pixi run run")
    print("  3. Save yourself with 's' key")
    print("  4. Move away and come back")
    print("  5. The agent will greet you and ask questions!")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
