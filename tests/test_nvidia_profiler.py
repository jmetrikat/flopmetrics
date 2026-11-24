import unittest
from unittest import mock

from flopmetrics import profiler


class EnergyCounterTest(unittest.TestCase):
    def test_window_energy_is_counter_difference(self):
        prof = profiler.NvidiaProfiler()
        prof._nvml_handles = ["gpu0"]  # set up by __enter__
        readings_mj = [0, 1_000, 3_000, 3_500, 6_500]
        with mock.patch.object(
            profiler.pynvml, "nvmlDeviceGetTotalEnergyConsumption", side_effect=readings_mj
        ):
            for step in ["other", "interest", "other", "interest", "other"]:
                prof.record_step(step)

        self.assertEqual(prof.get_total_energy(record_steps=["interest"], return_data=True), [2.0, 3.0])
        self.assertEqual(prof.get_total_energy(), 6.5)


if __name__ == "__main__":
    unittest.main()
