import unittest

from hand.cli import _force_bar, build_parser
from hand.client import _encode_int16, decode_int16


class HandClientTests(unittest.TestCase):
    def test_encode_minus_one_as_unsigned_register(self):
        self.assertEqual(_encode_int16(-1), 0xFFFF)

    def test_decode_signed_force_register(self):
        self.assertEqual(decode_int16(65473), -63)

    def test_force_bar_shows_sign(self):
        self.assertEqual(_force_bar(500, 1000, width=4), "    |##  ")
        self.assertEqual(_force_bar(-500, 1000, width=4), "  ##|    ")

    def test_cli_parses_six_angle_targets(self):
        args = build_parser().parse_args(["angle", "0", "1", "2", "3", "4", "1000"])
        self.assertEqual(args.targets, [0, 1, 2, 3, 4, 1000])

    def test_cli_rejects_out_of_range_angle(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["angle", "0", "0", "0", "0", "0", "1001"])

    def test_cli_parses_tactile_snapshot(self):
        args = build_parser().parse_args(["tactile", "--once", "--scale", "500"])
        self.assertTrue(args.once)
        self.assertEqual(args.scale, 500)


if __name__ == "__main__":
    unittest.main()
