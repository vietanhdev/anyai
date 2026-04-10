"""Tests for the enhanced anyai doctor CLI command (Iteration 42)."""

from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from anyai.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestDoctorCommand:
    """Tests for the enhanced 'doctor' command."""

    def test_doctor_runs_successfully(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0

    def test_doctor_shows_python_version(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "Python version" in result.output

    def test_doctor_shows_compute_device(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "Compute device" in result.output

    def test_doctor_shows_system_memory(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "System memory" in result.output

    def test_doctor_shows_disk_space(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "Disk" in result.output

    def test_doctor_shows_ollama_status(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "Ollama" in result.output

    def test_doctor_shows_installed_backends_table(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "Installed Backends" in result.output

    def test_doctor_shows_package_names(self, runner):
        result = runner.invoke(main, ["doctor"])
        # Should show at least some sub-package names
        assert "anycv" in result.output or "anyocr" in result.output or "anyllm" in result.output

    def test_doctor_shows_install_hints(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "pip install" in result.output

    def test_doctor_shows_system_health_title(self, runner):
        result = runner.invoke(main, ["doctor"])
        assert "System Health" in result.output

    def test_doctor_with_cuda_device(self, runner):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.total_mem = 8 * 1024**3
        mock_torch.cuda.memory_allocated.return_value = 1 * 1024**3
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "CUDA" in result.output

    def test_doctor_with_cpu_only(self, runner):
        with patch.dict("sys.modules", {"torch": None}):
            result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "CPU" in result.output
