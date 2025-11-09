"""
Main script demonstrating A2A interaction between bias interrogator and chat agent.

This script shows how two agents communicate:
1. Bias Interrogator (pydantic-ai) generates questions to detect bias
2. Chat Agent (Google A2A SDK) answers those questions
3. Bias Interrogator analyzes the responses for potential bias
"""

import asyncio
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from agents.bias_interrogator import BiasInterrogator, BiasQuestion
from agents.chat_agent import ChatAgent
from config import A2AConfig


class A2AOrchestrator:
    """
    Orchestrates the interaction between the bias interrogator and chat agent.
    """

    def __init__(
        self,
        config: Optional[A2AConfig] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        num_questions: Optional[int] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: A2AConfig object (preferred). If provided, overrides other args.
            model_name: Name of the Ollama model to use (legacy, for backward compatibility)
            base_url: Base URL for the Ollama API (legacy)
            num_questions: Number of bias-probing questions to generate (legacy)
        """
        # Use config if provided, otherwise use legacy parameters or defaults
        if config is None:
            config = A2AConfig.get_default()
            # Override with legacy parameters if provided
            if model_name:
                config.bias_interrogator.model.name = model_name
                config.chat_agent.model.name = model_name
            if base_url:
                config.bias_interrogator.model.base_url = base_url
                config.chat_agent.model.base_url = base_url
            if num_questions is not None:
                config.session.num_questions = num_questions

        self.config = config

        # Initialize agents from config
        self.interrogator = BiasInterrogator(
            model_name=config.bias_interrogator.model.name,
            base_url=config.bias_interrogator.model.base_url,
        )
        self.chat_agent = ChatAgent(
            agent_id=config.chat_agent.agent_id,
            model_name=config.chat_agent.model.name,
            base_url=config.chat_agent.model.base_url,
        )

        # Override system prompts if provided in config
        if config.bias_interrogator.system_prompt:
            self.interrogator.agent.system_prompt = config.bias_interrogator.system_prompt
        if config.chat_agent.system_prompt:
            self.chat_agent.system_prompt = config.chat_agent.system_prompt

        self.num_questions = config.session.num_questions
        self.verbose = config.session.verbose
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

    # Parse command line arguments
    config_file = None
    mode = None
    num_questions = None
    focus_area = None

    # Help message
    if "--help" in sys.argv or "-h" in sys.argv:
        print("\nUsage:")
        print("  python main.py [options]")
        print("\nOptions:")
        print("  -c, --config FILE    Load configuration from file (YAML, JSON, or TOML)")
        print("  -i, --interactive    Run in interactive mode")
        print("  -n NUM              Number of questions to generate (default: 5)")
        print("  -f FOCUS            Focus area for bias testing")
        print("  -h, --help          Show this help message")
        print("\nExamples:")
        print("  python main.py                           # Run with default config")
        print("  python main.py --config config.yaml      # Run with config file")
        print("  python main.py -n 10                     # Run with 10 questions")
        print("  python main.py -f gender                 # Focus on gender bias")
        print("  python main.py --interactive             # Run in interactive mode")
        print("  python main.py -c config.yaml -i         # Config file + interactive")
        return

    # Parse arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ["-c", "--config"]:
            if i + 1 < len(sys.argv):
                config_file = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --config requires a file path")
                return
        elif sys.argv[i] in ["-i", "--interactive"]:
            mode = "interactive"
            i += 1
        elif sys.argv[i] in ["-n", "--num"]:
            if i + 1 < len(sys.argv):
                num_questions = int(sys.argv[i + 1])
                i += 2
            else:
                print("Error: --num requires a number")
                return
        elif sys.argv[i] in ["-f", "--focus"]:
            if i + 1 < len(sys.argv):
                focus_area = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --focus requires an area")
                return
        else:
            i += 1

    # Load configuration
    config = None
    if config_file:
        try:
            config = A2AConfig.from_file(config_file)
            print(f"\nLoaded configuration from: {config_file}")
            print(f"  - Bias Interrogator Model: {config.bias_interrogator.model.name}")
            print(f"  - Chat Agent Model: {config.chat_agent.model.name}")
            print(f"  - Session Questions: {config.session.num_questions}")
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {config_file}")
            return
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return

        # Command line arguments override config file
        if mode:
            config.session.mode = mode
        if num_questions is not None:
            config.session.num_questions = num_questions
        if focus_area:
            config.session.focus_area = focus_area

        # Use mode and focus from config if not overridden
        if mode is None:
            mode = config.session.mode
        if focus_area is None:
            focus_area = config.session.focus_area
    else:
        # No config file, use defaults or CLI args
        if mode is None:
            mode = "auto"
        if num_questions is None:
            num_questions = 5

    # Create orchestrator
    if config:
        orchestrator = A2AOrchestrator(config=config)
    else:
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
