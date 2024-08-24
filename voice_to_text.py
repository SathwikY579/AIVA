import whisper
import webrtcvad
import pydub
from pydub import AudioSegment

# Load the Whisper model
model = whisper.load_model("base.en")

# Load audio
audio = AudioSegment.from_file("input_audio.wav", format="wav")
audio = audio.set_channels(1).set_frame_rate(16000)

# Apply VAD
vad = webrtcvad.Vad(0.5)
frames = pydub.utils.make_chunks(audio, 30)
speech_frames = [frame for frame in frames if vad.is_speech(frame.raw_data, 16000)]

# Convert speech to text
text = model.transcribe(speech_frames)
