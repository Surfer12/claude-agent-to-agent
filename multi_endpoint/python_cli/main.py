import argparse
import os


def run():
    """Simple Python CLI endpoint using a mock API key."""
    parser = argparse.ArgumentParser(description="Python endpoint")
    parser.add_argument("message", help="Message to echo")
    args = parser.parse_args()

    api_key = os.environ.get("MOCK_API_KEY", "mock-api-key")
    redacted = api_key[:4] + "***" if api_key else "(not set)"
    print(f"Python endpoint using API key: {redacted}")
    print(f"Message: {args.message}")


if __name__ == "__main__":
    run()
