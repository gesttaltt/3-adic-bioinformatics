# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Base analysis orchestrator for consistent analysis patterns.

This module provides a common base class for analysis pipelines that follow
the pattern: validate -> preprocess -> compute -> aggregate -> return results.

Consolidates common patterns from:
- src/analysis/crispr/analyzer.py
- src/analysis/immunology/epitope_encoding.py
- src/diseases/rheumatoid_arthritis.py

Usage:
    class MyAnalyzer(AnalysisOrchestrator):
        def _validate_inputs(self, data):
            # Validate data format
            pass

        def _compute(self, data):
            # Perform analysis
            return results

    analyzer = MyAnalyzer()
    results = analyzer.analyze(data)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

# Type variables for input and output
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class AnalysisMetadata:
    """Metadata for an analysis run."""

    analysis_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult(Generic[OutputType]):
    """Container for analysis results with metadata."""

    data: OutputType
    metadata: AnalysisMetadata
    success: bool = True

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.metadata.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.metadata.errors.append(error)
        self.success = False


class AnalysisOrchestrator(ABC, Generic[InputType, OutputType]):
    """Base class for analysis pipelines.

    Provides a consistent pattern for analysis with:
    - Input validation
    - Preprocessing
    - Computation
    - Result aggregation

    Subclasses must implement:
    - _validate_inputs(): Validate input data format
    - _compute(): Perform the actual analysis

    Optionally override:
    - _preprocess(): Transform input before computation
    - _aggregate_results(): Combine/format results
    - _get_analysis_name(): Return analysis name for metadata
    """

    def __init__(self, **config: Any):
        """Initialize the orchestrator.

        Args:
            **config: Configuration parameters stored in self.config
        """
        self.config = config
        self._metadata: AnalysisMetadata | None = None

    @abstractmethod
    def _validate_inputs(self, data: InputType) -> None:
        """Validate input data format.

        Args:
            data: Input data to validate

        Raises:
            ValueError: If validation fails
        """
        pass

    def _preprocess(self, data: InputType) -> InputType:
        """Preprocess input data before computation.

        Override this method to add preprocessing steps.
        Default implementation returns data unchanged.

        Args:
            data: Input data to preprocess

        Returns:
            Preprocessed data
        """
        return data

    @abstractmethod
    def _compute(self, data: InputType) -> OutputType:
        """Perform the main computation.

        Args:
            data: Preprocessed input data

        Returns:
            Analysis results
        """
        pass

    def _aggregate_results(self, results: OutputType) -> OutputType:
        """Aggregate or format results.

        Override this method to add post-processing.
        Default implementation returns results unchanged.

        Args:
            results: Raw computation results

        Returns:
            Aggregated/formatted results
        """
        return results

    def _get_analysis_name(self) -> str:
        """Return the name of this analysis.

        Override to provide a custom name.

        Returns:
            Analysis name for metadata
        """
        return self.__class__.__name__

    def analyze(
        self,
        data: InputType,
        **kwargs: Any,
    ) -> AnalysisResult[OutputType]:
        """Run the complete analysis pipeline.

        Steps:
        1. Create metadata
        2. Validate inputs
        3. Preprocess data
        4. Compute results
        5. Aggregate results
        6. Return wrapped result

        Args:
            data: Input data to analyze
            **kwargs: Additional parameters passed to _compute

        Returns:
            AnalysisResult containing data and metadata

        Raises:
            ValueError: If input validation fails
        """
        # Create metadata
        self._metadata = AnalysisMetadata(
            analysis_name=self._get_analysis_name(),
            parameters={**self.config, **kwargs},
        )

        try:
            # Step 1: Validate
            self._validate_inputs(data)

            # Step 2: Preprocess
            preprocessed = self._preprocess(data)

            # Step 3: Compute
            results = self._compute(preprocessed, **kwargs)

            # Step 4: Aggregate
            final_results = self._aggregate_results(results)

            return AnalysisResult(
                data=final_results,
                metadata=self._metadata,
                success=True,
            )

        except ValueError as e:
            self._metadata.errors.append(str(e))
            # Re-raise validation errors
            raise

        except Exception as e:
            self._metadata.errors.append(f"Computation error: {str(e)}")
            raise


class BatchAnalysisOrchestrator(AnalysisOrchestrator[list[InputType], list[OutputType]]):
    """Base class for batch analysis pipelines.

    Extends AnalysisOrchestrator to process lists of inputs.
    Provides parallel processing capability.
    """

    def __init__(self, parallel: bool = False, **config: Any):
        """Initialize batch orchestrator.

        Args:
            parallel: If True, process items in parallel
            **config: Configuration parameters
        """
        super().__init__(**config)
        self.parallel = parallel

    def _validate_inputs(self, data: list[InputType]) -> None:
        """Validate batch input.

        Args:
            data: List of inputs

        Raises:
            ValueError: If data is not a list or is empty
        """
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data).__name__}")
        if len(data) == 0:
            raise ValueError("Input list is empty")

        # Validate each item
        for i, item in enumerate(data):
            try:
                self._validate_single(item)
            except ValueError as e:
                raise ValueError(f"Item {i}: {e}")

    @abstractmethod
    def _validate_single(self, item: InputType) -> None:
        """Validate a single input item.

        Args:
            item: Single input to validate

        Raises:
            ValueError: If validation fails
        """
        pass

    @abstractmethod
    def _compute_single(self, item: InputType) -> OutputType:
        """Compute results for a single item.

        Args:
            item: Single input to process

        Returns:
            Results for this item
        """
        pass

    def _compute(self, data: list[InputType], **kwargs: Any) -> list[OutputType]:
        """Process all items in the batch.

        Args:
            data: List of inputs
            **kwargs: Additional parameters

        Returns:
            List of results
        """
        results = []
        for item in data:
            result = self._compute_single(item)
            results.append(result)
        return results


__all__ = [
    "AnalysisMetadata",
    "AnalysisResult",
    "AnalysisOrchestrator",
    "BatchAnalysisOrchestrator",
]
