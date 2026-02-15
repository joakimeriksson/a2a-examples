"""
Bias evaluation framework using A2A agent interaction.

The evaluator model generates bias-probing questions and counterfactual pairs,
sends them to the target model, and analyzes responses for bias.

All questions are LLM-generated — the evaluator is informed it's testing for bias
and creates targeted probes dynamically.

Supports Ollama (local) and OpenAI-compatible APIs.
"""

import argparse
import asyncio
import json
import os
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

from agents.bias_interrogator import (
    BiasInterrogator,
    BiasAnalysis,
    CounterfactualAnalysis,
    GeneratedCounterfactualPair,
)
from agents.chat_agent import ChatAgent, KNOWN_PROVIDERS


# --- Result dataclasses ---


@dataclass
class StandaloneResult:
    question: str
    category: str
    rationale: str
    response: str
    analysis: Dict[str, Any]
    run_index: int


@dataclass
class CounterfactualResult:
    template: str
    variable: str
    category: str
    label_a: str
    label_b: str
    question_a: str
    question_b: str
    response_a: str
    response_b: str
    analysis: Dict[str, Any]
    run_index: int


@dataclass
class EvaluationReport:
    target_model: str
    evaluator_model: str
    timestamp: str
    num_runs: int
    categories_tested: List[str]
    question_source: str  # "generated"
    standalone_results: List[Dict[str, Any]] = field(default_factory=list)
    counterfactual_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# --- Orchestrator ---


class BiasEvaluator:
    """Orchestrates structured bias evaluation of a target model."""

    def __init__(
        self,
        target_model: str = "gemma3:latest",
        evaluator_model: str = "gemma3:latest",
        target_base_url: str = "http://localhost:11434",
        evaluator_base_url: str = "http://localhost:11434",
        target_api_key: Optional[str] = None,
        target_backend: str = "ollama",
        num_runs: int = 1,
    ):
        self.target_model = target_model
        self.evaluator_model = evaluator_model
        self.num_runs = num_runs

        self.interrogator = BiasInterrogator(
            model_name=evaluator_model, base_url=evaluator_base_url
        )
        self.chat_agent = ChatAgent(
            agent_id="target-model",
            model_name=target_model,
            base_url=target_base_url,
            api_key=target_api_key,
            backend=target_backend,
        )

        self.standalone_results: List[StandaloneResult] = []
        self.counterfactual_results: List[CounterfactualResult] = []

    async def generate_and_run_standalone(
        self, num_questions: int = 10, focus_area: Optional[str] = None
    ) -> List[StandaloneResult]:
        """Generate questions with the evaluator, then test them on the target."""
        print(f"\n  Generating {num_questions} bias-probing questions...")
        questions = await self.interrogator.generate_questions(num_questions, focus_area)
        print(f"  Generated {len(questions.questions)} questions.\n")

        results = []
        for i, q in enumerate(questions.questions):
            for run in range(self.num_runs):
                run_label = f" (run {run + 1}/{self.num_runs})" if self.num_runs > 1 else ""
                print(f"  [{i + 1}/{len(questions.questions)}]{run_label} {q.category}: {q.question[:80]}...")

                self.chat_agent.reset_conversation()
                response = await self.chat_agent.process_message(q.question)

                analysis = await self.interrogator.analyze_response(
                    q.question, response.message, q.category
                )

                result = StandaloneResult(
                    question=q.question,
                    category=q.category,
                    rationale=q.rationale,
                    response=response.message,
                    analysis=analysis.model_dump(),
                    run_index=run,
                )
                results.append(result)

                severity = analysis.severity_score
                bias_flag = " [BIAS DETECTED]" if analysis.bias_detected else ""
                print(f"    Severity: {severity:.1f}/10{bias_flag}")

        self.standalone_results.extend(results)
        return results

    async def generate_and_run_counterfactual(
        self, num_pairs: int = 10, focus_area: Optional[str] = None
    ) -> List[CounterfactualResult]:
        """Generate counterfactual pairs with the evaluator, then test them."""
        print(f"\n  Generating {num_pairs} counterfactual pairs...")
        generated = await self.interrogator.generate_counterfactual_pairs(num_pairs, focus_area)
        print(f"  Generated {len(generated.pairs)} pairs.\n")

        results = []
        for i, pair in enumerate(generated.pairs):
            question_a = pair.template.replace(f"{{{pair.variable}}}", pair.variant_a_value)
            question_b = pair.template.replace(f"{{{pair.variable}}}", pair.variant_b_value)

            for run in range(self.num_runs):
                run_label = f" (run {run + 1}/{self.num_runs})" if self.num_runs > 1 else ""
                print(f"  [{i + 1}/{len(generated.pairs)}]{run_label} {pair.category}: {pair.variable} ({pair.variant_a_label} vs {pair.variant_b_label})")
                print(f"    Template: {pair.template[:80]}...")

                self.chat_agent.reset_conversation()
                response_a = await self.chat_agent.process_message(question_a)

                self.chat_agent.reset_conversation()
                response_b = await self.chat_agent.process_message(question_b)

                analysis = await self.interrogator.analyze_counterfactual(
                    template=pair.template,
                    variable=pair.variable,
                    label_a=pair.variant_a_label,
                    label_b=pair.variant_b_label,
                    response_a=response_a.message,
                    response_b=response_b.message,
                    category=pair.category,
                )

                result = CounterfactualResult(
                    template=pair.template,
                    variable=pair.variable,
                    category=pair.category,
                    label_a=pair.variant_a_label,
                    label_b=pair.variant_b_label,
                    question_a=question_a,
                    question_b=question_b,
                    response_a=response_a.message,
                    response_b=response_b.message,
                    analysis=analysis.model_dump(),
                    run_index=run,
                )
                results.append(result)

                severity = analysis.severity_score
                diff_flag = " [DIFFERENTIAL TREATMENT]" if analysis.differential_treatment else ""
                favored = f" (favors: {analysis.favored_variant})" if analysis.favored_variant else ""
                print(f"    Differential score: {severity:.1f}/10{diff_flag}{favored}")

        self.counterfactual_results.extend(results)
        return results

    def compute_summary(self) -> Dict[str, Any]:
        """Compute quantitative summary statistics."""
        summary: Dict[str, Any] = {
            "total_standalone_tests": len(self.standalone_results),
            "total_counterfactual_tests": len(self.counterfactual_results),
            "num_runs": self.num_runs,
        }

        if self.standalone_results:
            scores = [r.analysis["severity_score"] for r in self.standalone_results]
            bias_detected_count = sum(1 for r in self.standalone_results if r.analysis["bias_detected"])

            summary["standalone"] = {
                "mean_severity": round(statistics.mean(scores), 2),
                "median_severity": round(statistics.median(scores), 2),
                "max_severity": round(max(scores), 2),
                "min_severity": round(min(scores), 2),
                "stdev_severity": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
                "bias_detection_rate": round(bias_detected_count / len(self.standalone_results), 3),
                "total_biases_flagged": bias_detected_count,
            }

            categories: Dict[str, List[float]] = {}
            for r in self.standalone_results:
                categories.setdefault(r.category, []).append(r.analysis["severity_score"])

            summary["standalone"]["by_category"] = {
                cat: {
                    "mean_severity": round(statistics.mean(cat_scores), 2),
                    "count": len(cat_scores),
                }
                for cat, cat_scores in sorted(categories.items())
            }

        if self.counterfactual_results:
            diff_scores = [r.analysis["severity_score"] for r in self.counterfactual_results]
            diff_detected_count = sum(
                1 for r in self.counterfactual_results if r.analysis["differential_treatment"]
            )

            summary["counterfactual"] = {
                "mean_differential": round(statistics.mean(diff_scores), 2),
                "median_differential": round(statistics.median(diff_scores), 2),
                "max_differential": round(max(diff_scores), 2),
                "min_differential": round(min(diff_scores), 2),
                "stdev_differential": round(statistics.stdev(diff_scores), 2) if len(diff_scores) > 1 else 0.0,
                "differential_treatment_rate": round(diff_detected_count / len(self.counterfactual_results), 3),
                "total_differential_flagged": diff_detected_count,
            }

            categories = {}
            for r in self.counterfactual_results:
                categories.setdefault(r.category, []).append(r.analysis["severity_score"])

            summary["counterfactual"]["by_category"] = {
                cat: {
                    "mean_differential": round(statistics.mean(cat_scores), 2),
                    "count": len(cat_scores),
                }
                for cat, cat_scores in sorted(categories.items())
            }

            favored_counts: Dict[str, int] = {}
            for r in self.counterfactual_results:
                fav = r.analysis.get("favored_variant")
                if fav:
                    favored_counts[fav] = favored_counts.get(fav, 0) + 1
            summary["counterfactual"]["favored_variants"] = favored_counts

        all_scores = []
        if self.standalone_results:
            all_scores.extend(r.analysis["severity_score"] for r in self.standalone_results)
        if self.counterfactual_results:
            all_scores.extend(r.analysis["severity_score"] for r in self.counterfactual_results)

        summary["aggregate_bias_index"] = round(statistics.mean(all_scores), 2) if all_scores else 0.0

        return summary

    def build_report(self) -> EvaluationReport:
        """Build full evaluation report."""
        summary = self.compute_summary()
        return EvaluationReport(
            target_model=self.target_model,
            evaluator_model=self.evaluator_model,
            timestamp=datetime.utcnow().isoformat(),
            num_runs=self.num_runs,
            question_source="generated",
            categories_tested=sorted(
                set(
                    [r.category for r in self.standalone_results]
                    + [r.category for r in self.counterfactual_results]
                )
            ),
            standalone_results=[asdict(r) for r in self.standalone_results],
            counterfactual_results=[asdict(r) for r in self.counterfactual_results],
            summary=summary,
        )

    def print_report(self, summary: Dict[str, Any]):
        """Print formatted summary to terminal."""
        print("\n" + "=" * 80)
        print(f"BIAS EVALUATION REPORT: {self.target_model}")
        print("=" * 80)
        print(f"  Target model:    {self.target_model}")
        print(f"  Evaluator model: {self.evaluator_model}")
        print(f"  Runs per test:   {self.num_runs}")
        print(f"  Questions:       LLM-generated")
        print(f"  Aggregate bias index: {summary['aggregate_bias_index']:.2f} / 10.0")

        if "standalone" in summary:
            s = summary["standalone"]
            print("\n" + "-" * 80)
            print("STANDALONE QUESTION RESULTS")
            print("-" * 80)
            print(f"  Tests run:           {summary['total_standalone_tests']}")
            print(f"  Bias detection rate: {s['bias_detection_rate']:.1%}")
            print(f"  Mean severity:       {s['mean_severity']:.2f} / 10.0")
            print(f"  Median severity:     {s['median_severity']:.2f}")
            print(f"  Std deviation:       {s['stdev_severity']:.2f}")
            print(f"  Range:               {s['min_severity']:.1f} - {s['max_severity']:.1f}")

            print("\n  By category:")
            for cat, data in s["by_category"].items():
                bar = "█" * int(data["mean_severity"]) + "░" * (10 - int(data["mean_severity"]))
                print(f"    {cat:<16} {bar} {data['mean_severity']:.2f}  (n={data['count']})")

        if "counterfactual" in summary:
            c = summary["counterfactual"]
            print("\n" + "-" * 80)
            print("COUNTERFACTUAL PAIR RESULTS")
            print("-" * 80)
            print(f"  Tests run:                  {summary['total_counterfactual_tests']}")
            print(f"  Differential treatment rate: {c['differential_treatment_rate']:.1%}")
            print(f"  Mean differential:           {c['mean_differential']:.2f} / 10.0")
            print(f"  Median differential:         {c['median_differential']:.2f}")
            print(f"  Std deviation:               {c['stdev_differential']:.2f}")
            print(f"  Range:                       {c['min_differential']:.1f} - {c['max_differential']:.1f}")

            print("\n  By category:")
            for cat, data in c["by_category"].items():
                bar = "█" * int(data["mean_differential"]) + "░" * (10 - int(data["mean_differential"]))
                print(f"    {cat:<16} {bar} {data['mean_differential']:.2f}  (n={data['count']})")

            if c.get("favored_variants"):
                print("\n  Favored variants:")
                for variant, count in sorted(c["favored_variants"].items(), key=lambda x: -x[1]):
                    print(f"    {variant}: {count} times")

        print("\n" + "=" * 80)

    def export_json(self, filepath: str):
        """Export full results to JSON."""
        report = self.build_report()
        with open(filepath, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\nResults exported to: {filepath}")


# --- Comparison ---


def print_comparison(model_a: str, summary_a: Dict, model_b: str, summary_b: Dict):
    """Print side-by-side comparison of two models."""
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 80)

    idx_a = summary_a.get("aggregate_bias_index", 0)
    idx_b = summary_b.get("aggregate_bias_index", 0)
    print(f"\n  {'Metric':<30} {model_a:<25} {model_b:<25}")
    print(f"  {'-'*30} {'-'*25} {'-'*25}")
    print(f"  {'Aggregate bias index':<30} {idx_a:<25.2f} {idx_b:<25.2f}")

    if "standalone" in summary_a and "standalone" in summary_b:
        sa, sb = summary_a["standalone"], summary_b["standalone"]
        print(f"  {'Standalone mean severity':<30} {sa['mean_severity']:<25.2f} {sb['mean_severity']:<25.2f}")
        print(f"  {'Bias detection rate':<30} {sa['bias_detection_rate']:<25.1%} {sb['bias_detection_rate']:<25.1%}")

        all_cats = sorted(set(list(sa.get("by_category", {}).keys()) + list(sb.get("by_category", {}).keys())))
        if all_cats:
            print(f"\n  {'Category':<30} {model_a:<25} {model_b:<25}")
            print(f"  {'-'*30} {'-'*25} {'-'*25}")
            for cat in all_cats:
                val_a = sa.get("by_category", {}).get(cat, {}).get("mean_severity", "-")
                val_b = sb.get("by_category", {}).get(cat, {}).get("mean_severity", "-")
                va = f"{val_a:.2f}" if isinstance(val_a, (int, float)) else val_a
                vb = f"{val_b:.2f}" if isinstance(val_b, (int, float)) else val_b
                print(f"  {cat:<30} {va:<25} {vb:<25}")

    if "counterfactual" in summary_a and "counterfactual" in summary_b:
        ca, cb = summary_a["counterfactual"], summary_b["counterfactual"]
        print(f"\n  {'Counterfactual mean diff':<30} {ca['mean_differential']:<25.2f} {cb['mean_differential']:<25.2f}")
        print(f"  {'Differential treatment rate':<30} {ca['differential_treatment_rate']:<25.1%} {cb['differential_treatment_rate']:<25.1%}")

    if idx_a < idx_b:
        print(f"\n  Lower bias: {model_a} (by {idx_b - idx_a:.2f} points)")
    elif idx_b < idx_a:
        print(f"\n  Lower bias: {model_b} (by {idx_a - idx_b:.2f} points)")
    else:
        print(f"\n  Result: Tied")

    print("=" * 80)


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bias evaluation framework for AI models (LLM-generated questions)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Evaluate a local Ollama model:
  python main.py eval --target gemma3:latest

  # Focus on political bias:
  python main.py eval --target qwen2:latest --focus political

  # Compare two Ollama models:
  python main.py compare --model-a llama3:latest --model-b qwen2:latest

  # Compare with focus on political bias, 3 runs each:
  python main.py compare --model-a llama3:latest --model-b qwen2:latest \\
      --focus political --runs 3

  # Use a stronger evaluator model:
  python main.py eval --target qwen2:latest --evaluator gemma3:latest

  # Use cloud APIs:
  python main.py eval --target deepseek-chat --target-api-base deepseek

  # Interactive mode:
  python main.py interactive --target llama3:latest
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- eval ---
    eval_p = subparsers.add_parser("eval", help="Evaluate a single model for bias")
    _add_eval_args(eval_p)
    _add_target_args(eval_p)

    # --- compare ---
    cmp_p = subparsers.add_parser("compare", help="Compare two models head-to-head")
    _add_eval_args(cmp_p)
    cmp_p.add_argument("--model-a", required=True, help="First Ollama model (or model ID for cloud API)")
    cmp_p.add_argument("--api-base-a", default="http://localhost:11434", help="API base for model A")
    cmp_p.add_argument("--api-key-a", default=None, help="API key for model A")
    cmp_p.add_argument("--model-b", required=True, help="Second Ollama model (or model ID for cloud API)")
    cmp_p.add_argument("--api-base-b", default="http://localhost:11434", help="API base for model B")
    cmp_p.add_argument("--api-key-b", default=None, help="API key for model B")

    # --- interactive ---
    int_p = subparsers.add_parser("interactive", help="Interactive question mode")
    _add_target_args(int_p)

    return parser


def _add_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument("--evaluator", default="gemma3:latest", help="Evaluator model (default: gemma3:latest)")
    parser.add_argument("--evaluator-base-url", default="http://localhost:11434", help="Evaluator API base URL")
    parser.add_argument("--focus", "-f", default=None,
                        help="Focus area for generated questions (e.g., political, gender, race)")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Runs per question (default: 1)")
    parser.add_argument("--n-questions", "-n", type=int, default=10, help="Number of standalone questions to generate (default: 10)")
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of counterfactual pairs to generate (default: 10)")
    parser.add_argument("--pairs-only", action="store_true", help="Only run counterfactual pair tests")
    parser.add_argument("--standalone-only", action="store_true", help="Only run standalone question tests")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path")


def _add_target_args(parser: argparse.ArgumentParser):
    parser.add_argument("--target", default="gemma3:latest", help="Target Ollama model to evaluate (default: gemma3:latest)")
    parser.add_argument("--target-api-base", default="http://localhost:11434",
                        help=f"Target API base URL or shorthand ({', '.join(KNOWN_PROVIDERS.keys())})")
    parser.add_argument("--target-api-key", default=None, help="API key for target model")


def resolve_api_key(provided: Optional[str], provider_hint: str) -> Optional[str]:
    """Resolve API key from argument or environment variable."""
    if provided:
        return provided
    env_map = {
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = env_map.get(provider_hint)
    if env_var:
        return os.environ.get(env_var)
    return None


def make_evaluator(
    target_model: str,
    target_api_base: str,
    target_api_key: Optional[str],
    evaluator_model: str,
    evaluator_base_url: str,
    num_runs: int,
) -> BiasEvaluator:
    """Create a BiasEvaluator with resolved settings."""
    api_key = resolve_api_key(target_api_key, target_api_base)
    backend = "openai" if (api_key or target_api_base in KNOWN_PROVIDERS) else "ollama"

    return BiasEvaluator(
        target_model=target_model,
        evaluator_model=evaluator_model,
        target_base_url=target_api_base,
        evaluator_base_url=evaluator_base_url,
        target_api_key=api_key,
        target_backend=backend,
        num_runs=num_runs,
    )


async def run_evaluation(evaluator: BiasEvaluator, args, label: str = "") -> Dict[str, Any]:
    """Generate questions and run evaluation."""
    if not args.standalone_only:
        print(f"\n--- {label}Counterfactual pair testing ---")
        await evaluator.generate_and_run_counterfactual(args.n_pairs, args.focus)

    if not args.pairs_only:
        print(f"\n--- {label}Standalone question testing ---")
        await evaluator.generate_and_run_standalone(args.n_questions, args.focus)

    summary = evaluator.compute_summary()
    evaluator.print_report(summary)
    return summary


async def cmd_eval(args):
    """Evaluate a single model."""
    evaluator = make_evaluator(
        target_model=args.target,
        target_api_base=args.target_api_base,
        target_api_key=args.target_api_key,
        evaluator_model=args.evaluator,
        evaluator_base_url=args.evaluator_base_url,
        num_runs=args.runs,
    )

    print("\n" + "=" * 80)
    print("BIAS EVALUATION")
    print("=" * 80)
    print(f"  Target:    {args.target} ({evaluator.chat_agent.backend})")
    print(f"  Evaluator: {args.evaluator}")
    print(f"  Runs:      {args.runs}")
    print(f"  Questions: {args.n_questions} standalone + {args.n_pairs} pairs (LLM-generated)")
    if args.focus:
        print(f"  Focus:     {args.focus}")
    print("=" * 80)

    await run_evaluation(evaluator, args)

    output = args.output or f"bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.export_json(output)


async def cmd_compare(args):
    """Compare two models head-to-head."""
    evaluator_a = make_evaluator(
        target_model=args.model_a,
        target_api_base=args.api_base_a,
        target_api_key=args.api_key_a,
        evaluator_model=args.evaluator,
        evaluator_base_url=args.evaluator_base_url,
        num_runs=args.runs,
    )
    evaluator_b = make_evaluator(
        target_model=args.model_b,
        target_api_base=args.api_base_b,
        target_api_key=args.api_key_b,
        evaluator_model=args.evaluator,
        evaluator_base_url=args.evaluator_base_url,
        num_runs=args.runs,
    )

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD BIAS COMPARISON")
    print("=" * 80)
    print(f"  Model A:   {args.model_a} ({evaluator_a.chat_agent.backend})")
    print(f"  Model B:   {args.model_b} ({evaluator_b.chat_agent.backend})")
    print(f"  Evaluator: {args.evaluator}")
    print(f"  Runs:      {args.runs}")
    print(f"  Questions: {args.n_questions} standalone + {args.n_pairs} pairs (LLM-generated)")
    if args.focus:
        print(f"  Focus:     {args.focus}")
    print("=" * 80)

    print(f"\n{'='*80}")
    print(f"EVALUATING: {args.model_a}")
    print(f"{'='*80}")
    summary_a = await run_evaluation(evaluator_a, args, label=f"[{args.model_a}] ")

    print(f"\n{'='*80}")
    print(f"EVALUATING: {args.model_b}")
    print(f"{'='*80}")
    summary_b = await run_evaluation(evaluator_b, args, label=f"[{args.model_b}] ")

    print_comparison(args.model_a, summary_a, args.model_b, summary_b)

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.output.replace(".json", "") if args.output else f"bias_compare_{timestamp}"

    comparison_report = {
        "comparison_timestamp": datetime.utcnow().isoformat(),
        "model_a": {"name": args.model_a, "summary": summary_a},
        "model_b": {"name": args.model_b, "summary": summary_b},
        "model_a_full": asdict(evaluator_a.build_report()),
        "model_b_full": asdict(evaluator_b.build_report()),
    }
    filepath = f"{base}.json"
    with open(filepath, "w") as f:
        json.dump(comparison_report, f, indent=2, default=str)
    print(f"\nComparison exported to: {filepath}")


async def cmd_interactive(args):
    """Interactive mode."""
    evaluator = make_evaluator(
        target_model=args.target,
        target_api_base=args.target_api_base,
        target_api_key=args.target_api_key,
        evaluator_model="gemma3:latest",
        evaluator_base_url="http://localhost:11434",
        num_runs=1,
    )

    print("\n" + "=" * 80)
    print(f"INTERACTIVE MODE - Target: {args.target}")
    print("=" * 80)
    print("Ask questions and get structured bias analysis.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            evaluator.chat_agent.reset_conversation()
            response = await evaluator.chat_agent.process_message(question)
            print(f"\nResponse:\n{response.message}\n")

            analysis = await evaluator.interrogator.analyze_response(question, response.message)
            print(f"Bias detected: {analysis.bias_detected}")
            print(f"Severity: {analysis.severity_score:.1f}/10")
            if analysis.detected_biases:
                for b in analysis.detected_biases:
                    print(f"  - [{b.bias_type}] {b.evidence}")
                    print(f"    {b.explanation}")
            if analysis.positive_aspects:
                print(f"Positive: {', '.join(analysis.positive_aspects)}")
            print(f"Assessment: {analysis.overall_assessment}\n")

        except (KeyboardInterrupt, EOFError):
            break


async def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "eval":
            await cmd_eval(args)
        elif args.command == "compare":
            await cmd_compare(args)
        elif args.command == "interactive":
            await cmd_interactive(args)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
