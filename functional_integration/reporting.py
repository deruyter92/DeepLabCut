"""
Defines the TestResult class and utilities for summarizing and persisting
integration test results.

Credits: maximpavliv  https://github.com/DeepLabCut/DLC-benchmarking
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import json
import os
from typing import Any


class TestStatus(Enum):
    """Possible statuses for an integration test."""
    PASSED = "passed"
    FAILED = "failed"
    # SKIPPED = "skipped" # for now we don't skip tests


@dataclass
class TestResult:
    """Structured result for an integration test."""
    test_name: str
    status: TestStatus
    duration: float = 0.0
    notes: str | None = None
    extra: dict[str, Any] | None = None  # Optional free-form info (metrics, warnings, etc.)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        d = asdict(self)
        d["status"] = self.status.value  # Serialize Enum as string
        return d


def summarize_results(results: list[TestResult], output_dir: str = "integration_results"):
    """Print a formatted summary of all test results and write to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{timestamp}.json")

    print("\n=== Integration Test Summary ===")
    print(f"{'Test Name':<40} {'Status':<10} {'Duration (s)':>12}")
    print("-" * 65)

    for res in results:
        print(f"{res.test_name:<40} {res.status.value:<10} {res.duration:>12.1f}")

    with open(output_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    print("\nDetailed results written to:", output_path)


@dataclass
class IntegrationTestSummary:
    """
    Structured summary of an integration test’s detailed outcomes.

    This class is used to collect and report the results of a single
    integration test (e.g., inference comparison, training evaluation,
    tracking consistency test).

    Each integration test creates an `IntegrationTestSummary` instance,
    records the outcome of individual checks (e.g., one per video, model,
    or experiment configuration) using `add_result()`, and returns the
    summary object to the higher-level test runner.

    Attributes
    ----------
    test_name : str
        Name of the integration test (e.g., "superanimal_inference").
    passed : int
        Number of successful checks.
    failed : int
        Number of failed checks.
    missing_files : int
        Number of cases skipped or failed due to missing data.
    errors : list[str]
        Human-readable error messages.
    details : list[dict[str, Any]]
        Arbitrary per-check metadata (e.g., video name, model, config, metric values).

    Typical usage pattern
    ---------------------
    >>> summary = IntegrationTestSummary("pose_estimation_test")
    >>> summary.add_result(True, video="mouse1", model="resnet50", metric="OK")
    >>> summary.add_missing("Missing reference file", video="mouse2")
    >>> summary.add_error("Runtime error: CUDA OOM", video="mouse3")
    >>> print(summary.to_dict())
    {
        "test_name": "pose_estimation_test",
        "passed": 1,
        "failed": 1,
        "missing_files": 1,
        "errors": [...],
        "details": [...]
    }

    Each integration test defines what metadata to store in `details`.
    This keeps the summary structure generic and reusable across different
    types of DLC integration tests.
    """
    test_name: str
    passed: int = 0
    failed: int = 0
    missing_files: int = 0
    errors: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, passed: bool, **info: Any):
        """
        Add a single test case result.

        Parameters
        ----------
        passed : bool
            Whether this sub-test passed or failed.
        **info : dict
            Arbitrary metadata (e.g., video name, detector, training config, etc.)
        """
        entry = {"passed": passed, **info}
        self.details.append(entry)
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def add_missing(self, message: str, **info: Any):
        """Record a missing file or unavailable data case."""
        self.missing_files += 1
        self.errors.append(message)
        self.details.append({"passed": False, "missing": True, "message": message, **info})

    def add_error(self, message: str, **info: Any):
        """Record an unexpected runtime error."""
        self.failed += 1
        self.errors.append(message)
        self.details.append({"passed": False, "error": True, "message": message, **info})

    def to_dict(self):
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "failed": self.failed,
            "missing_files": self.missing_files,
            "errors": self.errors,
            "details": self.details,
        }

    def __add__(self, other: "IntegrationTestSummary") -> "IntegrationTestSummary":
        """
        Merge two IntegrationTestSummary objects.
        Combines counts, merges details, and concatenates names.
        """
        if not isinstance(other, IntegrationTestSummary):
            return NotImplemented

        merged = IntegrationTestSummary(
            test_name=f"{self.test_name} + {other.test_name}",
            passed=self.passed + other.passed,
            failed=self.failed + other.failed,
            missing_files=self.missing_files + other.missing_files,
            errors=self.errors + other.errors,
            details=self.details + other.details,
        )
        return merged

    def to_test_result(self, duration: float = 0.0) -> "TestResult":
        """
        Convert this summary into a TestResult object.
        Automatically determines pass/fail based on counts.
        """
        status = (
            TestStatus.PASSED
            if self.failed == 0 and self.missing_files == 0
            else TestStatus.FAILED
        )
        notes = (
            f"{self.passed} passed, {self.failed} failed, "
            f"{self.missing_files} missing"
        )
        return TestResult(
            test_name=self.test_name,
            status=status,
            duration=duration,
            notes=notes,
            extra={"summary": self.to_dict()},
        )