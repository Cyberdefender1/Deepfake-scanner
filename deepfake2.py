from models import load_video_model, load_audio_model
from preprocessors import preprocess_video, preprocess_audio
from behavioral_monitor import BehavioralMonitor

# Load pretrained models
video_model = load_video_model()
audio_model = load_audio_model()

# Initialize behavioral monitor
monitor = BehavioralMonitor()

# Function to detect deepfakes in video
def detect_video_deepfake(video_file):

    # Extract frames and preprocess data
    frame_data = preprocess_video(video_file)

    # Make predictions
    preds = video_model.predict(frame_data)

    # Determine if deepfake
    result = "Deepfake" if preds > 0.5 else "Real"

    # Track behavioral changes
    monitor.update(video_model.layers, video_file, result, preds)

    return result, preds[0][0]

# Function to detect deepfakes in audio
def detect_audio_deepfake(audio_file):

    # Extract audio segments and preprocess data
    audio_data = preprocess_audio(audio_file)

    # Make predictions
    preds = audio_model.predict(audio_data)

    # Determine if deepfake
    result = "Deepfake" if preds > 0.5 else "Real"

    # Track behavioral changes
    monitor.update(audio_model.layers, audio_file, result, preds)

    return result, preds[0][0]

# Test
video_result, video_prob = detect_video_deepfake("video.mp4")
audio_result, audio_prob = detect_audio_deepfake("audio.wav")

# View behavioral report
report = monitor.generate_report()
