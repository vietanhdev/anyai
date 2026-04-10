"""Pipeline/chain API for composing AI operations.

Provides sequential and parallel pipeline execution with named steps,
error handling, and debugging support.
"""

import concurrent.futures
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

# A step is either a bare callable or a (name, callable) tuple.
StepSpec = Union[Callable, Tuple[str, Callable]]


class PipelineStepError(Exception):
    """Raised when a pipeline step fails, identifying the step by name."""

    def __init__(self, step_name: str, step_index: int, original_error: Exception):
        self.step_name = step_name
        self.step_index = step_index
        self.original_error = original_error
        super().__init__(
            f"Pipeline failed at step {step_index} ({step_name!r}): "
            f"{type(original_error).__name__}: {original_error}"
        )


class Pipeline:
    """Chains callables sequentially, passing each output as input to the next.

    Each step can be a bare callable or a ``(name, callable)`` tuple.
    Named steps are useful for debugging: if a step fails, the error message
    identifies which step caused the problem.
    """

    def __init__(self, steps: Sequence[StepSpec]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step")
        self._steps: List[Tuple[str, Callable]] = []
        for i, step in enumerate(steps):
            if isinstance(step, tuple):
                name, fn = step
                if not callable(fn):
                    raise TypeError(
                        f"Step {i} ({name!r}): expected callable, got {type(fn).__name__}"
                    )
                self._steps.append((name, fn))
            elif callable(step):
                self._steps.append((f"step_{i}", step))
            else:
                raise TypeError(
                    f"Step {i}: expected callable or (name, callable) tuple, "
                    f"got {type(step).__name__}"
                )

    @property
    def steps(self) -> List[Tuple[str, Callable]]:
        """Return the list of (name, callable) step tuples."""
        return list(self._steps)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        name, fn = self._steps[0]
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            raise PipelineStepError(name, 0, exc) from exc

        for i, (name, fn) in enumerate(self._steps[1:], start=1):
            try:
                result = fn(result)
            except Exception as exc:
                raise PipelineStepError(name, i, exc) from exc

        return result

    def __repr__(self) -> str:
        step_names = [name for name, _ in self._steps]
        return f"Pipeline({step_names!r})"


class ParallelPipeline:
    """Runs multiple callables in parallel on the same input.

    Each branch is identified by a key. All branches receive the same
    input arguments and run concurrently using a thread pool.

    Returns a dictionary mapping each key to its branch result.
    """

    def __init__(self, branches: Dict[str, Callable]) -> None:
        if not branches:
            raise ValueError("ParallelPipeline requires at least one branch")
        for key, fn in branches.items():
            if not callable(fn):
                raise TypeError(
                    f"Branch {key!r}: expected callable, got {type(fn).__name__}"
                )
        self._branches = dict(branches)

    @property
    def branches(self) -> Dict[str, Callable]:
        """Return the dictionary of branch callables."""
        return dict(self._branches)

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        errors: Dict[str, Exception] = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_key = {
                executor.submit(fn, *args, **kwargs): key
                for key, fn in self._branches.items()
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    errors[key] = exc

        if errors:
            error_details = "; ".join(
                f"{key!r}: {type(exc).__name__}: {exc}"
                for key, exc in errors.items()
            )
            raise PipelineStepError(
                step_name=", ".join(errors.keys()),
                step_index=-1,
                original_error=RuntimeError(
                    f"Parallel pipeline failed in branches: {error_details}"
                ),
            )

        return results

    def __repr__(self) -> str:
        keys = list(self._branches.keys())
        return f"ParallelPipeline({keys!r})"


def _pipeline_factory(*steps: StepSpec) -> Pipeline:
    """Create a sequential pipeline from a list of steps.

    Each step can be a bare callable or a ``(name, callable)`` tuple.
    Steps can be passed as positional arguments or as a single sequence.

    Args:
        *steps: Callables or (name, callable) tuples.

    Returns:
        A :class:`Pipeline` instance.

    Example::

        from anyai import pipeline

        chain = pipeline(
            ("double", lambda x: x * 2),
            ("add_one", lambda x: x + 1),
        )
        result = chain(5)  # 11
    """
    # Support both pipeline([step1, step2]) and pipeline(step1, step2)
    if len(steps) == 1 and isinstance(steps[0], (list, tuple)) and not callable(steps[0]):
        steps = steps[0]
    return Pipeline(steps)


def _parallel_factory(branches: Dict[str, Callable]) -> ParallelPipeline:
    """Create a parallel pipeline that runs branches concurrently.

    All branches receive the same input and execute in parallel using
    a thread pool.

    Args:
        branches: A dictionary mapping branch names to callables.

    Returns:
        A :class:`ParallelPipeline` instance.

    Example::

        from anyai import pipeline

        par = pipeline.parallel({
            "doubled": lambda x: x * 2,
            "squared": lambda x: x ** 2,
        })
        result = par(5)  # {"doubled": 10, "squared": 25}
    """
    return ParallelPipeline(branches)


# Attach parallel as an attribute of the factory function so callers
# can write ``pipeline.parallel({...})``.
_pipeline_factory.parallel = _parallel_factory  # type: ignore[attr-defined]

pipeline = _pipeline_factory
