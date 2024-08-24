from edge_tts import TTS, VoicesManager

# Initialize TTS engine
tts = TTS(voice="en-US-JoannaNeural", pitch="high", rate="1.25")

# Convert text to speech
audio_output = tts.speak(output_text, "output_audio.mp3")
