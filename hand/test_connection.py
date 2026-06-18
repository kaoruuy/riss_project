"""Read-only connection check retained for compatibility."""

from hand.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["status"]))
