"""Standalone VAD → STT → TTS test using the production speech dispatcher."""

from flzat_speech.speech_ingress import main as run_speech


def main(args=None) -> None:
    run_speech(args, echo=True)


if __name__ == "__main__":
    main()
