"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every required section marked with TODO.
    2. Do NOT change class/function signatures. The optional ``contexts``
       parameter in ``run_full_eval`` is part of the required interface.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v

The reranking helper is an optional bonus exercise and may remain unimplemented.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------


@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """

    # Định nghĩa các trường dữ liệu cho QAPair
    question: str
    expected_answer: str
    context: str | None = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """

    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool = False
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0
        """
        # Tính trung bình cộng của 3 answer metrics (Faithfulness, Relevance, Completeness)
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------
# In production, replace with actual RAGAS framework:
#   from ragas import evaluate
#   from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
#
# Or DeepEval:
#   from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#   assert_test(test_case, [faithfulness, hallucination])
#
# Or TruLens:
#   from trulens.core import Feedback
#   f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
# ---------------------------------------------------------------------------

# Common English stopwords are ignored so overlap reflects *content* words,
# not filler (otherwise "is"/"a"/"the" inflate every score).
STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "as",
    "by",
    "and",
    "or",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "from",
    "into",
    "than",
}


_WORD_REGEX: re.Pattern[str] = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = _WORD_REGEX.findall(text.lower())
    return {t for t in tokens if t not in STOPWORDS}


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            answer_tokens = _tokenize(answer)
            context_tokens = _tokenize(context)
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.

        Returns:
            float in [0.0, 1.0] — 1.0 = fully grounded in context.
        """
        # Tách token cho answer; nếu answer rỗng thì mặc định 1.0 (không vi phạm groundedness)
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0

        # Tách token cho context và tính tỷ lệ token answer xuất hiện trong context
        context_tokens = _tokenize(context)
        overlap = len(answer_tokens & context_tokens)
        score = overlap / len(answer_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.

        Returns:
            float in [0.0, 1.0]
        """
        # Tách token cho question; nếu question rỗng thì mặc định 1.0
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0

        # Tách token cho answer và tính tỷ lệ token question được trả lời
        answer_tokens = _tokenize(answer)
        overlap = len(answer_tokens & question_tokens)
        score = overlap / len(question_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Returns:
            float in [0.0, 1.0]
        """
        # Tách token cho expected answer; nếu expected rỗng thì mặc định 1.0
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0

        # Tách token cho answer và đo mức độ bao phủ thông tin chuẩn
        answer_tokens = _tokenize(answer)
        overlap = len(answer_tokens & expected_tokens)
        score = overlap / len(expected_tokens)
        return max(0.0, min(1.0, float(score)))

    # -----------------------------------------------------------------------
    # Task 2b — Retrieval-side metrics (evaluate the GET-CONTEXT step)
    # -----------------------------------------------------------------------
    # From lecture (RAG pipeline): Context Recall → Context Precision →
    #   Faithfulness → Answer Relevancy. The two below score the RETRIEVER,
    #   operating on a LIST of chunks (order = retriever rank).
    # -----------------------------------------------------------------------

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — how much of the expected answer is covered by the
        UNION of retrieved chunks.

        Heuristic:
            union_tokens = ⋃ _tokenize(chunk) for chunk in contexts
            recall = |expected_tokens ∩ union_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.

        Low recall => retriever missed evidence the answer needs.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        # Hợp nhất tất cả các token từ các context chunk được retrieve
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)

        # Tính tỷ lệ token của expected answer xuất hiện trong tập hợp ngữ cảnh
        overlap = len(expected_tokens & union_tokens)
        score = overlap / len(expected_tokens)
        return max(0.0, min(1.0, float(score)))

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — RANK-AWARE Average Precision (AP@K), like RAGAS.
        Rewards retrievers that place RELEVANT chunks BEFORE noise.

        Steps:
            1. A chunk is "relevant" if it covers >= relevance_threshold of the
               expected tokens:  |chunk ∩ expected| / |expected| >= threshold
            2. Precision@k = (#relevant in top-k) / k
            3. AP@K = (1 / #relevant) * Σ_k [ Precision@k · relevant_k ]

        Return 1.0 if expected empty; 0.0 if no chunks or none relevant.
        Reordering relevant chunks earlier (reranking) raises this score.
        """
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        relevant_so_far = 0
        precision_sum = 0.0
        total_relevant = 0

        # Duyệt theo thứ tự xếp hạng (rank k từ 1 đến K)
        for k, chunk in enumerate(contexts, start=1):
            chunk_tokens = _tokenize(chunk)
            overlap = len(chunk_tokens & expected_tokens)
            coverage = overlap / len(expected_tokens)
            is_relevant = 1 if coverage >= relevance_threshold else 0

            if is_relevant:
                relevant_so_far += 1
                precision_at_k = relevant_so_far / k
                precision_sum += precision_at_k
                total_relevant += 1

        if total_relevant == 0:
            return 0.0

        ap_at_k = precision_sum / total_relevant
        return max(0.0, min(1.0, float(ap_at_k)))

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
        contexts: list[str] | None = None,
    ) -> EvalResult:
        """
        Run the three answer-side evaluations and, when ``contexts`` is
        supplied, both retrieval-side evaluations.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"

        Retrieval wiring:
            contexts is None → context_recall and context_precision stay None
            contexts provided → evaluate and store both retrieval metrics

        The two retrieval metrics diagnose the retriever and do not change the
        three-metric ``passed`` rule or ``overall_score()``.

        Returns:
            EvalResult with all fields populated.
        """
        # 1. Tính toán 3 answer metrics
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        # 2. Kiểm tra điều kiện passed (cả 3 metrics >= 0.5)
        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        # 3. Xác định failure_type nếu không pass (ưu tiên theo thứ tự first match wins)
        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        # 4. Tính toán retrieval metrics nếu contexts được cung cấp
        context_recall: float | None = None
        context_precision: float | None = None
        retrieved: list[str] = []

        if contexts is not None:
            context_recall = self.evaluate_context_recall(contexts, expected)
            context_precision = self.evaluate_context_precision(contexts, expected)
            retrieved = contexts

        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
            retrieved_contexts=retrieved,
        )

        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
            context_precision=context_precision,
            context_recall=context_recall,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------


def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """A minimal lexical reranker: sort chunks by word overlap with the query,
    most-overlapping first. Stand-in for a real cross-encoder reranker.

    Reordering relevant chunks toward the top increases the rank-aware
    Context Precision WITHOUT changing the retrieved set.

    Hint: sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)),
                 reverse=True)
    """
    # Sắp xếp các chunk theo số lượng token trùng khớp với query giảm dần
    query_tokens = _tokenize(query)
    return sorted(
        contexts,
        key=lambda c: len(_tokenize(c) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------
# From lecture:
#   - Judge LLM nhận: question + agent answer + reference answer + rubric
#   - Judge trả về: Score 1-5 + Rationale
#   - Best practices: multiple judges, randomize order, calibrate against human
#   - Biases: positional, verbosity, self-preference
#   - Rubric template:
#       5 = Correct, complete, well-cited
#       4 = Mostly correct, minor gaps
#       3 = Partially correct, some errors
#       2 = Significant errors or missing info
#       1 = Wrong or irrelevant
# ---------------------------------------------------------------------------


class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        # Lưu trữ hàm callable mô phỏng hoặc gọi Judge LLM
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Args:
            question: The original question.
            answer:   The AI's answer to score.
            rubric:   Dict mapping criterion name → description.
                      Example: {"accuracy": "Is the answer factually correct?",
                                "clarity": "Is the answer clear and well-structured?"}

        Behavior:
            1. Build a judge prompt that includes the question, answer, and rubric.
            2. Call judge_llm_fn(prompt).
            3. Parse the response for scores.

        For simplicity, if the LLM response can't be parsed as JSON scores,
        return a default score of 0.5 for each criterion.

        Returns:
            {
                "scores":    dict[str, float],  # criterion → score 0-1
                "reasoning": str,               # raw LLM explanation
            }
        """
        # 1. Xây dựng prompt đánh giá bao gồm câu hỏi, câu trả lời và rubric
        prompt = (
            f"You are an expert judge evaluating AI responses.\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Rubric: {json.dumps(rubric, ensure_ascii=False)}\n\n"
            f"Score each criterion on a scale of 0.0 to 1.0 in JSON format."
        )

        # 2. Gọi hàm judge_llm_fn
        raw_response = self.judge_llm_fn(prompt)
        scores: dict[str, float] = {}
        reasoning = raw_response

        # 3. Phân tích kết quả JSON trả về
        try:
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else json.loads(raw_response)
            if isinstance(parsed, dict):
                # Trường hợp cấu trúc có trường "scores"
                if "scores" in parsed and isinstance(parsed["scores"], dict):
                    scores = {k: float(v) for k, v in parsed["scores"].items()}
                    reasoning = str(parsed.get("reasoning", raw_response))
                else:
                    # Trường hợp trả về trực tiếp dict {criterion: score}
                    for criterion in rubric:
                        if criterion in parsed:
                            scores[criterion] = float(parsed[criterion])
                        else:
                            scores[criterion] = 0.5
                    for k, v in parsed.items():
                        if k not in scores and isinstance(v, (int, float)):
                            scores[k] = float(v)
            else:
                scores = {criterion: 0.5 for criterion in rubric}
        except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError):
            scores = {criterion: 0.5 for criterion in rubric}

        return {
            "scores": scores,
            "reasoning": reasoning,
        }

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Checks:
            positional_bias: Check if first response consistently scores higher
            leniency_bias:   Average score > 0.8 across all criteria
            severity_bias:   Average score < 0.3 across all criteria

        Args:
            scores_batch: List of score dicts from score_response().

        Returns:
            {
                "positional_bias": bool,
                "leniency_bias":   bool,
                "severity_bias":   bool,
            }
        """
        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }

        # Thu thập toàn bộ điểm số của tất cả các tiêu chí trong batch
        all_scores: list[float] = []
        item_averages: list[float] = []

        for item in scores_batch:
            item_score_vals = [float(v) for v in item.get("scores", {}).values()]
            if item_score_vals:
                item_averages.append(sum(item_score_vals) / len(item_score_vals))
                all_scores.extend(item_score_vals)

        # 1. Leniency bias: Điểm trung bình > 0.8 (chấm quá nương tay)
        # 2. Severity bias: Điểm trung bình < 0.3 (chấm quá khắt khe)
        if all_scores:
            batch_avg = sum(all_scores) / len(all_scores)
            leniency_bias = batch_avg > 0.8
            severity_bias = batch_avg < 0.3
        else:
            leniency_bias = False
            severity_bias = False

        # 3. Positional bias: Response đầu tiên luôn nhận điểm cao hơn các response sau
        positional_bias = False
        if len(item_averages) >= 2:
            first_avg = item_averages[0]
            rest_avg = sum(item_averages[1:]) / len(item_averages[1:])
            positional_bias = first_avg > rest_avg

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------
# From lecture:
#   - CI/CD integration: Framework + CI/CD = quality gate tự động
#   - Agent với faithfulness < 0.7 → không được deploy
#   - Regression = metric drop > 0.05 vs baseline
#   - Triggers: mỗi code release, mỗi prompt change, trước demo/launch
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """
        Run all QA pairs through the agent and evaluate each result.

        Args:
            qa_pairs:   List of QAPair objects.
            agent_fn:   Function str → str (the agent's answer function).
            evaluator:  RAGASEvaluator instance.

        Returns:
            List of EvalResult, one per qa_pair.
        """
        results: list[EvalResult] = []
        for pair in qa_pairs:
            # 1. Sinh câu trả lời từ agent
            actual_answer = agent_fn(pair.question)

            # 2. Chạy đánh giá full pipeline với retrieval contexts nếu có
            contexts = pair.retrieved_contexts if pair.retrieved_contexts else None
            eval_res = evaluator.run_full_eval(
                answer=actual_answer,
                question=pair.question,
                context=pair.context or "",
                expected=pair.expected_answer,
                contexts=contexts,
            )

            # 3. Giữ nguyên đối tượng QAPair gốc trên EvalResult để bảo toàn metadata
            eval_res.qa_pair = pair
            results.append(eval_res)

        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """
        Generate an aggregate report from evaluation results.

        Returns:
            {
                "total":            int,
                "passed":           int,
                "pass_rate":        float,  # passed / total
                "avg_faithfulness": float,
                "avg_relevance":    float,
                "avg_completeness": float,
                "avg_context_recall": float | None,
                "avg_context_precision": float | None,
                "failure_types":    dict[str, int],  # type → count
            }

        Average only non-None retrieval scores. Return None for a retrieval
        average when no result contains that metric.
        """
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "avg_context_recall": None,
                "avg_context_precision": None,
                "failure_types": {},
            }

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / total

        avg_f = sum(r.faithfulness for r in results) / total
        avg_r = sum(r.relevance for r in results) / total
        avg_c = sum(r.completeness for r in results) / total

        # Tính trung bình các điểm retrieval (chỉ tính các kết quả không phải None)
        recall_vals = [
            r.context_recall for r in results if r.context_recall is not None
        ]
        avg_recall = (sum(recall_vals) / len(recall_vals)) if recall_vals else None

        precision_vals = [
            r.context_precision for r in results if r.context_precision is not None
        ]
        avg_precision = (
            (sum(precision_vals) / len(precision_vals)) if precision_vals else None
        )

        # Đếm các loại lỗi cho các trường hợp không pass
        failure_types: dict[str, int] = {}
        for r in results:
            if not r.passed and r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed_count,
            "pass_rate": pass_rate,
            "avg_faithfulness": avg_f,
            "avg_relevance": avg_r,
            "avg_completeness": avg_c,
            "avg_context_recall": avg_recall,
            "avg_context_precision": avg_precision,
            "failure_types": failure_types,
        }

    def run_regression(
        self,
        new_results: list[EvalResult],
        baseline_results: list[EvalResult],
    ) -> dict[str, Any]:
        """Compare new evaluation results against a baseline.

        A regression is when a metric's average drops by more than 0.05 vs baseline.

        Args:
            new_results: List of EvalResult instances (current run)
            baseline_results: List of EvalResult instances (reference/baseline)

        Returns:
            dict with keys:
              - 'new_avg_faithfulness': float
              - 'new_avg_relevance': float
              - 'new_avg_completeness': float
              - 'baseline_avg_faithfulness': float
              - 'baseline_avg_relevance': float
              - 'baseline_avg_completeness': float
              - 'regressions': list[str] — names of metrics that regressed
              - 'passed': bool — True if no regressions
        """
        new_f = (
            sum(r.faithfulness for r in new_results) / len(new_results)
            if new_results
            else 0.0
        )
        new_r = (
            sum(r.relevance for r in new_results) / len(new_results)
            if new_results
            else 0.0
        )
        new_c = (
            sum(r.completeness for r in new_results) / len(new_results)
            if new_results
            else 0.0
        )

        base_f = (
            sum(r.faithfulness for r in baseline_results) / len(baseline_results)
            if baseline_results
            else 0.0
        )
        base_r = (
            sum(r.relevance for r in baseline_results) / len(baseline_results)
            if baseline_results
            else 0.0
        )
        base_c = (
            sum(r.completeness for r in baseline_results) / len(baseline_results)
            if baseline_results
            else 0.0
        )

        # Kiểm tra tụt điểm quá 0.05 cho từng metric
        regressions: list[str] = []
        if (base_f - new_f) > 0.05:
            regressions.append("faithfulness")
        if (base_r - new_r) > 0.05:
            regressions.append("relevance")
        if (base_c - new_c) > 0.05:
            regressions.append("completeness")

        passed = len(regressions) == 0

        return {
            "new_avg_faithfulness": new_f,
            "new_avg_relevance": new_r,
            "new_avg_completeness": new_c,
            "baseline_avg_faithfulness": base_f,
            "baseline_avg_relevance": base_r,
            "baseline_avg_completeness": base_c,
            "regressions": regressions,
            "passed": passed,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """
        Return EvalResults where any score is below threshold.

        Args:
            results:   Full list of EvalResults.
            threshold: Minimum acceptable score for any metric.

        Returns:
            List of failing EvalResults.
        """
        # Lọc các kết quả có bất kỳ metric nào rớt dưới threshold
        return [
            r
            for r in results
            if r.faithfulness < threshold
            or r.relevance < threshold
            or r.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------
# From lecture:
#   Failure Taxonomy:
#     - hallucination: bịa thông tin → faithfulness guardrail yếu
#     - irrelevant: không giải quyết câu hỏi → prompt ambiguous
#     - incomplete: bỏ sót thông tin → context window nhỏ, retrieval thiếu
#     - off_topic: trả lời chủ đề khác → intent detection sai
#     - refusal: từ chối khi nên trả lời → guardrails quá chặt
#
#   5 Whys Method: hỏi "Tại sao?" liên tục cho đến root cause
#   Failure Clustering: fix 1 root cause giải quyết nhiều failures cùng lúc
#   Continuous Improvement: Evaluate → Analyze → Improve → Augment → Repeat
# ---------------------------------------------------------------------------


class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(self, failures: list[EvalResult]) -> dict[str, int]:
        """
        Count failures by failure_type.

        Returns:
            dict mapping failure_type → count.
            Example: {"hallucination": 3, "irrelevant": 2, "incomplete": 5}
        """
        counts: dict[str, int] = {}
        for f in failures:
            ftype = f.failure_type or "unknown"
            counts[ftype] = counts.get(ftype, 0) + 1
        return counts

    def find_root_cause(self, failure: EvalResult) -> str:
        """
        Suggest a root cause for a single failure based on its scores.

        Returns one of these strings based on which score is lowest:
            "Context is missing or irrelevant — improve retrieval"
            "Answer does not address the question — improve prompt clarity"
            "Answer is missing key information — increase context window or improve generation"
            "Multiple issues detected — review full pipeline"
        """
        scores = [
            (failure.faithfulness, "faithfulness"),
            (failure.relevance, "relevance"),
            (failure.completeness, "completeness"),
        ]
        min_score = min(s[0] for s in scores)
        min_metrics = [s[1] for s in scores if abs(s[0] - min_score) < 1e-6]

        # Nếu có từ 2 metric cùng thấp nhất hoặc tất cả đều dưới ngưỡng nguy hiểm (< 0.3)
        if len(min_metrics) > 1 or (
            failure.faithfulness < 0.3
            and failure.relevance < 0.3
            and failure.completeness < 0.3
        ):
            return "Multiple issues detected — review full pipeline"

        lowest = min_metrics[0]
        if lowest == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        elif lowest == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        else:
            return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_log(
        self,
        failures: list[EvalResult],
        suggestions: list[str],
    ) -> str:
        """Generate a Markdown table logging failures and improvement actions.

        Format:
        | Failure ID | Type | Root Cause | Suggested Fix | Status |
        |------------|------|------------|---------------|--------|
        | F001       | ...  | ...        | ...           | Open   |

        Args:
            failures: List of EvalResult instances where passed=False
            suggestions: List of suggestion strings (one per failure, can be shorter list)

        Returns:
            Markdown table string with a row per failure. Status is always "Open".
        """
        lines = [
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |",
            "|------------|------|------------|---------------|--------|",
        ]
        for i, failure in enumerate(failures):
            fid = f"F{i + 1:03d}"
            ftype = failure.failure_type or "Unknown"
            rc = self.find_root_cause(failure)
            fix = suggestions[i] if i < len(suggestions) else "Investigate pipeline"
            lines.append(f"| {fid} | {ftype} | {rc} | {fix} | Open |")

        return "\n".join(lines)

    def generate_improvement_suggestions(self, failures: list[EvalResult]) -> list[str]:
        """
        Generate a prioritized list of improvement suggestions based on failure patterns.

        Each suggestion should be a concrete, actionable string.

        Examples:
            "Increase chunk size in RAG pipeline to reduce context fragmentation"
            "Add few-shot examples showing complete answers to improve completeness"
            "Implement hallucination checker to filter unsupported claims"

        Returns:
            List of at least 3 suggestion strings (or fewer if failures is empty).
        """
        if not failures:
            return []

        categories = self.categorize_failures(failures)
        suggestions: list[str] = []

        if "hallucination" in categories or any(f.faithfulness < 0.5 for f in failures):
            suggestions.append(
                "Implement hallucination checker to filter unsupported claims"
            )
        if "irrelevant" in categories or any(f.relevance < 0.5 for f in failures):
            suggestions.append(
                "Refine system prompt and add query reformulation to improve answer relevance"
            )
        if "incomplete" in categories or any(f.completeness < 0.5 for f in failures):
            suggestions.append(
                "Increase chunk size in RAG pipeline to reduce context fragmentation"
            )
        if "off_topic" in categories:
            suggestions.append(
                "Implement intent classification to filter out-of-domain and off-topic queries"
            )

        # Đảm bảo luôn trả về ít nhất 3 gợi ý hợp lệ
        fallback_pool = [
            "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "Add few-shot examples showing complete answers to improve completeness",
            "Implement hallucination checker to filter unsupported claims",
            "Deploy reranking step (cross-encoder) to promote relevant chunks to the top",
            "Establish CI/CD quality gate with regression testing on the golden dataset",
        ]
        for s in fallback_pool:
            if s not in suggestions:
                suggestions.append(s)
            if len(suggestions) >= 3:
                break

        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample golden dataset (mini version — use 20 pairs in actual lab)
    # From lecture: stratified sampling = 5 Easy + 7 Medium + 5 Hard + 3 Adversarial
    qa_pairs = [
        # Easy — factual lookup
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        # Medium — multi-step reasoning
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        # Hard — ambiguous
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        # Adversarial — out-of-scope
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        """Simple mock agent for testing. Replace with your actual agent."""
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    # Run benchmark
    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Identify and analyze failures
    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    # Categorize (from lecture: cluster before fix)
    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    # Root cause for each failure (from lecture: 5 Whys)
    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    # Improvement suggestions (from lecture: continuous improvement loop)
    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    # Generate improvement log (Markdown table)
    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
