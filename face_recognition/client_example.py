"""
Example client showing how to interact with the Face Recognition Agent HTTP server.
"""

import httpx
import asyncio
import json


class FaceRecognitionClient:
    """Client for interacting with Face Recognition Agent HTTP server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the Face Recognition Agent server
        """
        self.base_url = base_url.rstrip('/')

    async def health_check(self):
        """Check if the server is healthy."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            return response.json()

    async def list_people(self):
        """List all known people."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/people")
            return response.json()

    async def get_person(self, name: str):
        """
        Get information about a specific person.

        Args:
            name: Person's name

        Returns:
            Person information
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/person/{name}")
            return response.json()

    async def request_questions(self, questions: list):
        """
        Request the agent to collect specific information.

        Args:
            questions: List of question keys to collect

        Returns:
            Response from the agent
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/query",
                json={
                    "operation": "request_questions",
                    "questions": questions
                }
            )
            return response.json()

    async def search_people(self, query: dict):
        """
        Search for people by metadata.

        Args:
            query: Dictionary of metadata fields to match

        Returns:
            Search results
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search",
                json={"query": query}
            )
            return response.json()


async def demo_client():
    """Demonstration of client usage."""
    print("\n" + "=" * 80)
    print("FACE RECOGNITION CLIENT - DEMO")
    print("=" * 80)

    # Create client
    client = FaceRecognitionClient()

    # 1. Health check
    print("\n1. Checking server health...")
    try:
        health = await client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Known people: {health['people_count']}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure the server is running: python server.py")
        return

    # 2. List all people
    print("\n2. Listing all known people...")
    result = await client.list_people()
    print(f"   Found {result.get('count', 0)} people:")
    for person in result.get('people', []):
        print(f"     - {person}")

    # 3. Request additional questions
    print("\n3. Requesting food preference questions...")
    result = await client.request_questions([
        "favorite_food",
        "dietary_restrictions",
        "allergies"
    ])
    print(f"   Response: {result.get('message', 'Unknown')}")
    print(f"   Pending questions: {result.get('pending_questions', [])}")

    # 4. Get person details (if any exist)
    people = result.get('people', [])
    if people:
        print(f"\n4. Getting details for '{people[0]}'...")
        person_info = await client.get_person(people[0])
        if person_info.get('status') == 'success':
            person = person_info['person']
            print(f"   Name: {person['name']}")
            print(f"   First seen: {person['first_seen']}")
            print(f"   Last seen: {person['last_seen']}")
            print(f"   Metadata: {json.dumps(person['metadata'], indent=4)}")
        else:
            print(f"   Not found: {person_info.get('message', 'Unknown error')}")
    else:
        print("\n4. No people in database yet.")
        print("   Run the face recognition agent first to add people:")
        print("   python face_recognition_agent.py")

    # 5. Search for people (example)
    print("\n5. Searching for people who like 'Pizza'...")
    result = await client.search_people({"favorite_food": "Pizza"})
    print(f"   Found {result.get('count', 0)} matches:")
    for person in result.get('results', []):
        print(f"     - {person['name']}: {person.get('metadata', {})}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80 + "\n")


async def interactive_client():
    """Interactive client for manual testing."""
    print("\n" + "=" * 80)
    print("FACE RECOGNITION CLIENT - INTERACTIVE MODE")
    print("=" * 80)

    client = FaceRecognitionClient()

    # Test connection
    try:
        health = await client.health_check()
        print(f"\nConnected to server: {health['status']}")
        print(f"Known people: {health['people_count']}")
    except Exception as e:
        print(f"\nError connecting to server: {e}")
        print("Make sure the server is running: python server.py")
        return

    print("\nCommands:")
    print("  list              - List all people")
    print("  get <name>        - Get person information")
    print("  search <key=val>  - Search people by metadata")
    print("  request <q1,q2>   - Request questions to collect")
    print("  quit              - Exit")

    while True:
        try:
            command = input("\n> ").strip()

            if not command:
                continue

            if command == "quit":
                break

            elif command == "list":
                result = await client.list_people()
                print(f"\nKnown people ({result.get('count', 0)}):")
                for person in result.get('people', []):
                    print(f"  - {person}")

            elif command.startswith("get "):
                name = command[4:].strip()
                result = await client.get_person(name)
                print(f"\n{json.dumps(result, indent=2)}")

            elif command.startswith("search "):
                query_str = command[7:].strip()
                try:
                    key, value = query_str.split('=', 1)
                    query = {key.strip(): value.strip()}
                    result = await client.search_people(query)
                    print(f"\n{json.dumps(result, indent=2)}")
                except ValueError:
                    print("Invalid search format. Use: search key=value")

            elif command.startswith("request "):
                questions_str = command[8:].strip()
                questions = [q.strip() for q in questions_str.split(',')]
                result = await client.request_questions(questions)
                print(f"\n{json.dumps(result, indent=2)}")

            else:
                print("Unknown command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_client()
    else:
        await demo_client()


if __name__ == "__main__":
    asyncio.run(main())
