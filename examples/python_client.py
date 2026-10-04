from src.client import OpenSpeechClient


if __name__ == "__main__":
    # The server uses a self-signed HTTPS certificate by default; ssl_verify=False accepts it.
    client = OpenSpeechClient(base_url="https://localhost:8100", ssl_verify=False)
    print("Client ready. Example speak call:")
    audio = client.speak("Hello from Open Speech", voice="alloy", speed=1.0, response_format="wav")
    with open("example_output.wav", "wb") as f:
        f.write(audio)
    print("Wrote example_output.wav")
