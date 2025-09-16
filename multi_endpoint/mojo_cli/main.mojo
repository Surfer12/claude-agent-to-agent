from python import os, sys

fn main() -> None:
    let api_key = os.getenv("MOCK_API_KEY") or "mock-api-key"
    if sys.len(sys.argv) < 2:
        print("Please provide a message to echo")
        return
    print(f"Mojo endpoint using API key: {api_key}")
    print(f"Message: {sys.argv[1]}")
