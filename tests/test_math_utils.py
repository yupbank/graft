"""Unit tests for the math used in the oracle sanity check.

These tests use small arrays and do not require a GPU or model weights.
"""

import mlx.core as mx
import numpy as np


def _logsumexp(x: mx.array) -> mx.array:
    c = x.max()
    return c + mx.log(mx.exp(x - c).sum())


def _log_softmax(x: mx.array) -> mx.array:
    return x - _logsumexp(x)


def _kl_divergence(log_p: mx.array, log_q: mx.array) -> float:
    p = mx.exp(log_p)
    kl = mx.sum(p * (log_p - log_q))
    return kl.item()


class TestLogsumexp:
    def test_simple(self):
        x = mx.array([1.0, 2.0, 3.0])
        result = _logsumexp(x).item()
        expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
        assert abs(result - expected) < 1e-5

    def test_large_values(self):
        x = mx.array([1000.0, 1001.0, 1002.0])
        result = _logsumexp(x).item()
        assert np.isfinite(result)
        assert abs(result - 1002.4076) < 0.01


class TestLogSoftmax:
    def test_sums_to_one(self):
        x = mx.array([1.0, 2.0, 3.0])
        log_p = _log_softmax(x)
        total = mx.exp(log_p).sum().item()
        assert abs(total - 1.0) < 1e-5

    def test_preserves_order(self):
        x = mx.array([1.0, 3.0, 2.0])
        log_p = _log_softmax(x)
        assert mx.argmax(log_p).item() == 1


class TestKLDivergence:
    def test_identical_distributions(self):
        log_p = _log_softmax(mx.array([1.0, 2.0, 3.0]))
        kl = _kl_divergence(log_p, log_p)
        assert abs(kl) < 1e-6

    def test_different_distributions(self):
        log_p = _log_softmax(mx.array([1.0, 2.0, 3.0]))
        log_q = _log_softmax(mx.array([3.0, 2.0, 1.0]))
        kl = _kl_divergence(log_p, log_q)
        assert kl > 0

    def test_oracle_delta_is_zero(self):
        """When ft = raw, the delta is zero and recovered = large exactly."""
        s_large = mx.array([1.0, 5.0, 2.0, 0.5])
        s_raw = s_large
        s_ft = s_large
        score = s_large + 1.0 * (s_ft - s_raw)
        log_p_ft = _log_softmax(s_ft)
        log_p_recovered = _log_softmax(score)
        kl = _kl_divergence(log_p_ft, log_p_recovered)
        assert abs(kl) < 1e-7


class TestImport:
    def test_graft_import(self):
        import graft

        assert graft.__version__ == "0.1.0"
