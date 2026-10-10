import unittest
import tempfile
from pathlib import Path

from prefill_tail_analysis import analyze, chunks, log_histogram, merge_tail, scratch


class TailAnalysisTest(unittest.TestCase):
    def test_actual_default_and_disabled_long_chunk(self):
        self.assertEqual(chunks(2049), ([2049], 8192))
        self.assertEqual(chunks(8193), ([8192, 1], 8192))
        self.assertEqual(chunks(2049, long_chunk=0), ([2048, 1], 2048))
        self.assertEqual(chunks(8193, long_chunk=0), ([2048, 2048, 2048, 2048, 1], 2048))
        self.assertEqual(chunks(8193, context=9000), ([8192, 1], 8192))

    def test_preserves_tokens_and_only_merges_bounded_last_tail(self):
        for n in range(20000):
            plan, _ = chunks(n)
            merged = merge_tail(plan)
            self.assertEqual(sum(merged), n)
            if plan != merged:
                self.assertTrue(0 < plan[-1] <= 256)
                self.assertEqual(merged[:-1], plan[:-2])
                self.assertEqual(len(plan)-len(merged), 1)
        self.assertEqual(merge_tail([8192, 257]), [8192, 257])

    def test_scratch_rounding_boundaries(self):
        self.assertEqual(scratch(8193)["buffers"]["gdnprep"]-scratch(8192)["buffers"]["gdnprep"],
                         48*576*4)
        self.assertEqual(scratch(8193)["buffers"]["bscores"], scratch(8192)["buffers"]["bscores"])
        self.assertEqual(scratch(8193)["host_bytes"]-scratch(8192)["host_bytes"], 2560*4+64)
        self.assertEqual(analyze(8193)["cancellation_polls_removed"], 1)
        for name in ("bscores", "vis", "ascores", "ublk", "umask", "ucount", "qsh", "qsl"):
            self.assertEqual(scratch(8192)["buffers"][name], scratch(8448)["buffers"][name])

    def test_checkpoint_aggregate_is_not_an_eligible_call(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"cache.log"
            path.write_text("cache: prompt 8194 reused 0 prefilled 8193 ckpt 1\n"
                            "cache: prompt 8194 reused 0 prefilled 8193 ckpt 0\n"
                            "cache: prompt 8193 reused 0 prefilled 8192 ckpt 0\n"
                            "cache: prompt 100 reused 0 prefilled 99 ckpt 1\n")
            result = log_histogram([path])
        self.assertEqual(result["checkpoint_split_unknown"], 1)
        self.assertEqual(result["known_eligible"], 1)
        self.assertEqual(result["known_ineligible"], 2)

    def test_invalid_context_fails(self):
        with self.assertRaises(ValueError):
            chunks(8193, context=8192)
        with self.assertRaises(ValueError):
            chunks(10, base=0)


if __name__ == "__main__":
    unittest.main()
