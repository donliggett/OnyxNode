"""OnyxNode: a black-box flight recorder for AI agent runs.

Record every tool call, model call, decision and cost of a run; replay runs;
diff two runs and find the first step where they diverge; turn recorded runs
into a regression suite.
"""
__version__ = "0.1.0"

from .diff import diff_runs, first_divergence, format_report  # noqa: E402
from .recorder import (HttpSink, JsonlSink, LLMResult, MemorySink, Recorder,  # noqa: E402
                       ReplayedError, ReplayMiss, Run)
from .suite import add_case, run_case, run_suite  # noqa: E402

__all__ = ["Recorder", "Run", "LLMResult", "ReplayMiss", "ReplayedError",
           "MemorySink", "JsonlSink", "HttpSink",
           "diff_runs", "first_divergence", "format_report",
           "add_case", "run_case", "run_suite", "__version__"]
