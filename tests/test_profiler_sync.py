import unittest
from unittest import mock

from flopmetrics.profiler import Profiler, torch


class RecordStepSyncTest(unittest.TestCase):
    def test_waits_for_gpu_before_timestamp(self):
        profiler = Profiler()
        with (
            mock.patch.object(torch.cuda, "is_initialized", return_value=True),
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            synchronize.side_effect = lambda: self.assertEqual(len(profiler.record_steps), 1)
            profiler.record_step("interest")
        synchronize.assert_called_once()
        self.assertEqual(len(profiler.record_steps), 2)


if __name__ == "__main__":
    unittest.main()
