from src.client import OpenSpeechClient


if __name__ == "__main__":
    client = OpenSpeechClient(base_url="http://localhost:8100")
    print("Client ready. Example speak call:")
    audio = client.speak("Hello from Open Speech", voice="alloy", speed=1.0, response_format="wav")
    with open("example_output.wav", "wb") as f:
        f.write(audio)
    print("Wrote example_output.wav")
