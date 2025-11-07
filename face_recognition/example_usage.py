"""
Example usage demonstrating the Face Recognition Agent.

This script shows various ways to use the face recognition agent:
1. Interactive face recognition from webcam
2. A2A protocol queries from other agents
3. Dynamic question collection
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_recognition.face_recognition_agent import FaceRecognitionAgent


async def demo_basic_recognition():
    """Demo: Basic face recognition loop."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Face Recognition")
    print("=" * 80)
    print("\nThis demo will:")
    print("  1. Open your webcam")
    print("  2. Detect and recognize faces")
    print("  3. Allow you to add new people by pressing 's'")
    print("  4. Press 'q' to quit")
    print("\nStarting in 3 seconds...")
    await asyncio.sleep(3)

    agent = FaceRecognitionAgent(data_dir="face_recognition/people_data")
    await agent.process_recognition_loop(duration=0)  # Continuous


async def demo_a2a_queries():
    """Demo: A2A protocol queries."""
    print("\n" + "=" * 80)
    print("DEMO 2: A2A Protocol Queries")
    print("=" * 80)

    agent = FaceRecognitionAgent(data_dir="face_recognition/people_data")

    # List all people
    print("\n1. Listing all known people...")
    result = await agent.handle_a2a_request({"operation": "list_people"})
    print(f"Result: {json.dumps(result, indent=2)}")

    # Query specific person (if any exist)
    people = agent.db.list_people()
    if people:
        print(f"\n2. Querying information about '{people[0]}'...")
        result = await agent.handle_a2a_request({
            "operation": "query_person",
            "name": people[0]
        })
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print("\n2. No people in database yet. Run demo 1 first to add people.")

    # Request additional questions
    print("\n3. Requesting additional questions to be collected...")
    result = await agent.handle_a2a_request({
        "operation": "request_questions",
        "questions": ["favorite_food", "favorite_color", "hobby"]
    })
    print(f"Result: {json.dumps(result, indent=2)}")


async def demo_agent_to_agent():
    """Demo: Simulated agent-to-agent interaction."""
    print("\n" + "=" * 80)
    print("DEMO 3: Agent-to-Agent Interaction Simulation")
    print("=" * 80)

    # Create face recognition agent
    face_agent = FaceRecognitionAgent(data_dir="face_recognition/people_data")

    # Simulate a "preferences agent" requesting information
    print("\n[Preferences Agent] Requesting face recognition agent to collect food preferences...")

    request = {
        "operation": "request_questions",
        "questions": ["favorite_food", "dietary_restrictions", "allergies"],
        "requesting_agent": "preferences-agent"
    }

    response = await face_agent.handle_a2a_request(request)
    print(f"\n[Face Recognition Agent] Response:")
    print(json.dumps(response, indent=2))

    print("\n[System] The face recognition agent will now ask these questions")
    print("        when it encounters a new person or sees a known person.")
    print(f"\n[System] Current pending questions: {list(face_agent.pending_questions)}")

    # Simulate another agent querying for people
    print("\n[Preferences Agent] Querying all known people...")

    list_request = {"operation": "list_people"}
    list_response = await face_agent.handle_a2a_request(list_request)
    print(f"\n[Face Recognition Agent] Response:")
    print(json.dumps(list_response, indent=2))

    # If there are people, query their information
    if list_response.get("people"):
        person_name = list_response["people"][0]
        print(f"\n[Preferences Agent] Querying details about '{person_name}'...")

        query_request = {
            "operation": "query_person",
            "name": person_name
        }
        query_response = await face_agent.handle_a2a_request(query_request)
        print(f"\n[Face Recognition Agent] Response:")
        print(json.dumps(query_response, indent=2))


async def main():
    """Main menu for demos."""
    print("\n" + "=" * 80)
    print("FACE RECOGNITION AGENT - EXAMPLE USAGE")
    print("=" * 80)
    print("\nAvailable demos:")
    print("  1. Basic face recognition (webcam)")
    print("  2. A2A protocol queries")
    print("  3. Agent-to-agent interaction simulation")
    print("  4. Run all demos")
    print("  5. Exit")

    while True:
        choice = input("\nSelect demo (1-5): ").strip()

        if choice == "1":
            await demo_basic_recognition()
        elif choice == "2":
            await demo_a2a_queries()
        elif choice == "3":
            await demo_agent_to_agent()
        elif choice == "4":
            await demo_a2a_queries()
            print("\n" + "=" * 80)
            await demo_agent_to_agent()
            print("\n" + "=" * 80)
            print("\nNow starting webcam demo...")
            print("Press 'q' when done to continue.")
            await demo_basic_recognition()
        elif choice == "5":
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
