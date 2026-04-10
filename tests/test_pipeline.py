"""Tests for anyai.pipeline module."""

import pytest

from anyai.pipeline import (
    Pipeline,
    ParallelPipeline,
    PipelineStepError,
    pipeline,
)


class TestPipelineSequential:
    """Tests for the sequential Pipeline."""

    def test_single_step(self):
        p = pipeline([lambda x: x * 2])
        assert p(5) == 10

    def test_multiple_steps(self):
        p = pipeline([
            lambda x: x * 2,
            lambda x: x + 1,
            lambda x: x * 3,
        ])
        assert p(5) == 33  # (5*2 + 1) * 3

    def test_named_steps(self):
        p = pipeline([
            ("double", lambda x: x * 2),
            ("add_one", lambda x: x + 1),
        ])
        assert p(5) == 11

    def test_mixed_named_and_unnamed(self):
        p = pipeline([
            ("double", lambda x: x * 2),
            lambda x: x + 1,
        ])
        assert p(5) == 11

    def test_first_step_receives_kwargs(self):
        def start(a, b=0):
            return a + b

        p = pipeline([
            ("start", start),
            ("double", lambda x: x * 2),
        ])
        assert p(3, b=7) == 20

    def test_string_pipeline(self):
        p = pipeline([
            ("upper", lambda s: s.upper()),
            ("strip", lambda s: s.strip()),
            ("split", lambda s: s.split()),
        ])
        assert p("  hello world  ") == ["HELLO", "WORLD"]

    def test_steps_property(self):
        fn1 = lambda x: x
        fn2 = lambda x: x
        p = pipeline([("a", fn1), ("b", fn2)])
        steps = p.steps
        assert len(steps) == 2
        assert steps[0][0] == "a"
        assert steps[1][0] == "b"

    def test_auto_naming(self):
        p = pipeline([lambda x: x, lambda x: x])
        steps = p.steps
        assert steps[0][0] == "step_0"
        assert steps[1][0] == "step_1"

    def test_repr(self):
        p = pipeline([("a", lambda x: x), ("b", lambda x: x)])
        assert repr(p) == "Pipeline(['a', 'b'])"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one step"):
            pipeline([])

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="expected callable"):
            pipeline([("bad", 42)])

    def test_bare_non_callable_raises(self):
        with pytest.raises(TypeError, match="expected callable"):
            pipeline([42])


class TestPipelineErrorHandling:
    """Tests for pipeline error reporting."""

    def test_first_step_failure(self):
        def fail(x):
            raise ValueError("boom")

        p = pipeline([("explode", fail)])
        with pytest.raises(PipelineStepError) as exc_info:
            p(1)
        err = exc_info.value
        assert err.step_name == "explode"
        assert err.step_index == 0
        assert isinstance(err.original_error, ValueError)
        assert "boom" in str(err)

    def test_middle_step_failure(self):
        def fail(x):
            raise RuntimeError("oops")

        p = pipeline([
            ("ok", lambda x: x + 1),
            ("fail_here", fail),
            ("never", lambda x: x),
        ])
        with pytest.raises(PipelineStepError) as exc_info:
            p(1)
        err = exc_info.value
        assert err.step_name == "fail_here"
        assert err.step_index == 1

    def test_error_chains_original(self):
        def fail(x):
            raise KeyError("missing")

        p = pipeline([("step", fail)])
        with pytest.raises(PipelineStepError) as exc_info:
            p(1)
        assert exc_info.value.__cause__ is not None


class TestParallelPipeline:
    """Tests for ParallelPipeline."""

    def test_basic_parallel(self):
        par = pipeline.parallel({
            "doubled": lambda x: x * 2,
            "squared": lambda x: x ** 2,
        })
        result = par(5)
        assert result == {"doubled": 10, "squared": 25}

    def test_parallel_with_string(self):
        par = pipeline.parallel({
            "upper": lambda s: s.upper(),
            "length": lambda s: len(s),
        })
        result = par("hello")
        assert result == {"upper": "HELLO", "length": 5}

    def test_parallel_error(self):
        def fail(x):
            raise ValueError("nope")

        par = pipeline.parallel({
            "ok": lambda x: x,
            "bad": fail,
        })
        with pytest.raises(PipelineStepError) as exc_info:
            par(1)
        assert "bad" in str(exc_info.value)

    def test_parallel_empty_raises(self):
        with pytest.raises(ValueError, match="at least one branch"):
            pipeline.parallel({})

    def test_parallel_non_callable_raises(self):
        with pytest.raises(TypeError, match="expected callable"):
            pipeline.parallel({"bad": 42})

    def test_parallel_repr(self):
        par = pipeline.parallel({"a": lambda x: x, "b": lambda x: x})
        r = repr(par)
        assert "ParallelPipeline" in r
        assert "a" in r
        assert "b" in r

    def test_parallel_branches_property(self):
        fn1 = lambda x: x
        fn2 = lambda x: x
        par = pipeline.parallel({"a": fn1, "b": fn2})
        branches = par.branches
        assert set(branches.keys()) == {"a", "b"}


class TestPipelineImportFromAnyai:
    """Test that pipeline is accessible from the anyai top-level package."""

    def test_import_pipeline(self):
        from anyai import pipeline as p
        assert callable(p)

    def test_import_pipeline_parallel(self):
        from anyai import pipeline as p
        assert callable(p.parallel)

    def test_import_classes(self):
        from anyai import Pipeline, ParallelPipeline, PipelineStepError
        assert Pipeline is not None
        assert ParallelPipeline is not None
        assert PipelineStepError is not None


class TestVersionInfo:
    """Tests for anyai.version_info()."""

    def test_returns_dict(self):
        from anyai import version_info
        result = version_info()
        assert isinstance(result, dict)

    def test_contains_anyai(self):
        from anyai import version_info
        result = version_info()
        assert "anyai" in result
        import re
        assert re.match(r"\d+\.\d+\.\d+", result["anyai"])

    def test_no_missing_packages_crash(self):
        from anyai import version_info
        # Should not raise even if sub-packages are not installed.
        result = version_info()
        assert isinstance(result, dict)
