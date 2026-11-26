import pytest
from rclpy.qos import HistoryPolicy, ReliabilityPolicy

from aeb_common.qos import stream_qos


def test_profile_fields():
    qos = stream_qos("reliable", 10)
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.history == HistoryPolicy.KEEP_LAST
    assert qos.depth == 10


def test_unknown_reliability_fails_loudly():
    with pytest.raises(KeyError):
        stream_qos("lossy", 1)
