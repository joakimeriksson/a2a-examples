"""
Main script demonstrating A2A interaction between bias interrogator and chat agent.

This script shows how two agents communicate:
1. Bias Interrogator (pydantic-ai) generates questions to detect bias
2. Chat Agent (Google A2A SDK) answers those questions
3. Bias Interrogator analyzes the responses for potential bias
"""

import asyncio
import sys
from typing import List, Dict, Any
from agents.bias_interrogator import BiasInterrogator, BiasQuestion
from agents.chat_agent import ChatAgent


class A2AOrchestrator:
    """
    Orchestrates the interaction between the bias interrogator and chat agent.
    """

    def __init__(
        self,
        model_name: str = "gemma3:latest",
        base_url: str = "http://localhost:11434",
        num_questions: int = 5,
    ):
        """
        Initialize the orchestrator.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
            num_questions: Number of bias-probing questions to generate
        """
        self.interrogator = BiasInterrogator(model_name=model_name, base_url=base_url)
        self.chat_agent = ChatAgent(
            agent_id="chat-agent-1", model_name=model_name, base_url=base_url
        )
        self.num_questions = num_questions
        self.results: List[Dict[str, Any]] = []

    async def run_bias_test(self, focus_area: str = None) -> List[Dict[str, Any]]:
        """
        Run a complete bias testing session.

        Args:
            focus_area: Optional specific area of bias to focus on

        Returns:
            List of results containing questions, responses, and analyses
        """
        print("\n" + "=" * 100)
        print("A2A BIAS TESTING SESSION")
        print("=" * 100)
        print(f"\nBias Interrogator: Generating {self.num_questions} questions...")
        if focus_area:
            print(f"Focus area: {focus_area}")

        # Step 1: Generate bias-probing questions
        questions = await self.interrogator.generate_questions(
            num_questions=self.num_questions, focus_area=focus_area
        )

        print(f"Generated {len(questions.questions)} questions.\n")

        # Step 2: Ask each question to the chat agent and analyze responses
        for i, q in enumerate(questions.questions, 1):
            print("-" * 100)
            print(f"\n[Question {i}/{len(questions.questions)}]")
            print(f"Category: {q.category}")
            print(f"Question: {q.question}")
            print(f"Rationale: {q.rationale}")

            # Get response from chat agent
            print("\nChat Agent: Processing question...")
            response = await self.chat_agent.process_message(q.question)

            print(f"\nChat Agent Response:\n{response.message}")

            # Analyze response for bias
            print("\nBias Interrogator: Analyzing response...")
            analysis = await self.interrogator.ask_question(q.question, response.message)

            print(f"\nBias Analysis:\n{analysis}")

            # Store results
            result = {
                "question_number": i,
                "category": q.category,
                "question": q.question,
                "rationale": q.rationale,
                "response": response.message,
                "analysis": analysis,
                "timestamp": response.timestamp,
            }
            self.results.append(result)

            print("\n" + "-" * 100)

        return self.results

    def print_summary(self):
        """Print a summary of the bias testing session."""
        print("\n" + "=" * 100)
        print("BIAS TESTING SUMMARY")
        print("=" * 100)

        print(f"\nTotal questions asked: {len(self.results)}")

        # Count by category
        categories = {}
        for result in self.results:
            cat = result["category"]
            categories[cat] = categories.get(cat, 0) + 1

        print("\nQuestions by category:")
        for cat, count in categories.items():
            print(f"  - {cat}: {count}")

        print("\n" + "=" * 100)

    async def interactive_mode(self):
        """
        Run in interactive mode where user can ask their own questions.
        """
        print("\n" + "=" * 100)
        print("INTERACTIVE MODE")
        print("=" * 100)
        print("\nYou can now ask questions to the Chat Agent.")
        print("The Bias Interrogator will analyze each response.")
        print("Type 'quit' or 'exit' to end the session.\n")

        while True:
            try:
                question = input("Your question: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("\nEnding interactive session...")
                    break

                if not question:
                    continue

                # Get response from chat agent
                print("\nChat Agent: Processing...")
                response = await self.chat_agent.process_message(question)
                print(f"\nChat Agent Response:\n{response.message}")

                # Analyze for bias
                print("\nBias Interrogator: Analyzing response...")
                analysis = await self.interrogator.ask_question(question, response.message)
                print(f"\nBias Analysis:\n{analysis}\n")

            except (KeyboardInterrupt, EOFError):
                print("\n\nEnding interactive session...")
                break


async def main():
    """Main entry point."""
    print("\n" + "=" * 100)
    print("A2A AGENTS EXAMPLE: BIAS INTERROGATION")
    print("=" * 100)
    print("\nThis example demonstrates two agents communicating:")
    print("  1. Bias Interrogator (pydantic-ai)")
    print("  2. Chat Agent (Google A2A SDK)")
    print("\nBoth agents are backed by local Ollama models (gemma3:latest)")
    print("=" * 100)

    # Check command line arguments
    mode = "auto"
    num_questions = 5
    focus_area = None

    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            mode = "interactive"
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("\nUsage:")
            print("  python main.py [options]")
            print("\nOptions:")
            print("  -i, --interactive    Run in interactive mode")
            print("  -n NUM              Number of questions to generate (default: 5)")
            print("  -f FOCUS            Focus area for bias testing")
            print("  -h, --help          Show this help message")
            print("\nExamples:")
            print("  python main.py                    # Run automated test with 5 questions")
            print("  python main.py -n 10              # Run with 10 questions")
            print("  python main.py -f gender          # Focus on gender bias")
            print("  python main.py --interactive      # Run in interactive mode")
            return

    # Parse other arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ["-n", "--num"]:
            if i + 1 < len(sys.argv):
                num_questions = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        elif sys.argv[i] in ["-f", "--focus"]:
            if i + 1 < len(sys.argv):
                focus_area = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        else:
            i += 1

    # Create orchestrator
    orchestrator = A2AOrchestrator(num_questions=num_questions)

    try:
        if mode == "interactive":
            await orchestrator.interactive_mode()
        else:
            # Run automated bias test
            await orchestrator.run_bias_test(focus_area=focus_area)
            orchestrator.print_summary()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\nSession complete.")


if __name__ == "__main__":
    asyncio.run(main())
