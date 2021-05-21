import os
import time
import logging
import streamlit as st
import numpy as np
import pandas as pd
import librosa, librosa.display
import matplotlib.pyplot as plt
import pyaudio, wave, pylab
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


logger = logging.getLogger(__name__)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_DIR_AUDIO = os.path.join(DATA_DIR, 'audio')
DATA_DIR_GUITAR = os.path.join(DATA_DIR_AUDIO, 'Guitar_Only/')
DATA_DIR_AUGMENTED = os.path.join(DATA_DIR_AUDIO, 'augmented')

METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
METADATA_DIR_RAW = os.path.join(METADATA_DIR, 'raw')
METADATA_DIR_PROCESSED = os.path.join(METADATA_DIR, 'processed')

METADATA_DIR_AUGMENTED = os.path.join(METADATA_DIR, 'augmented')
METADATA_DIR_AUGMENTED_RAW = os.path.join(METADATA_DIR_AUGMENTED, 'raw')
METADATA_DIR_AUGMENTED_PROCESSED = os.path.join(METADATA_DIR_AUGMENTED, 'processed')

LOG_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_DIR_TRAINING = os.path.join(LOG_DIR, 'training')

OUT_DIR = os.path.join(ROOT_DIR, 'output/')
RECORDING_DIR = os.path.join(OUT_DIR, 'recording')
IMAGE_DIR = os.path.join(OUT_DIR, 'images')

WAVE_OUTPUT_FILE = os.path.join(RECORDING_DIR, "recorded.wav")
SPECTROGRAM_FILE = os.path.join(RECORDING_DIR, "spectrogram.png")

# Features #################
CLASSES = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g']
CLASSES_MAP = {'a':0, 'am':1, 'bm':2, 'c':3, 'd':4, 'dm':5, 'e':6, 'em':7, 'f':8, 'g':9}

# Audio configurations
INPUT_DEVICE = 0
MAX_INPUT_CHANNELS = 1  # Max input channels
DEFAULT_SAMPLE_RATE = 44100   # Default sample rate of microphone or recording device
DURATION = 3   # 3 seconds
CHUNK_SIZE = 1024

class Sound(object):
    def __init__(self):
        # Set default configurations for recording device
        # sd.default.samplerate = DEFAULT_SAMPLE_RATE
        # sd.default.channels = DEFAULT_CHANNELS
        self.format = pyaudio.paInt16
        self.channels = MAX_INPUT_CHANNELS
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.chunk = CHUNK_SIZE
        self.duration = DURATION
        self.path = WAVE_OUTPUT_FILE
        self.device = INPUT_DEVICE
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.device_info()
        print()
        logger.info("Audio device configurations currently used")
        logger.info(f"Default input device index = {self.device}")
        logger.info(f"Max input channels = {self.channels}")
        logger.info(f"Default samplerate = {self.sample_rate}")

    def device_info(self):
        num_devices = self.audio.get_device_count()
        keys = ['name', 'index', 'maxInputChannels', 'defaultSampleRate']
        logger.info(f"List of System's Audio Devices configurations:")
        logger.info(f"Number of audio devices: {num_devices}")
        for i in range(num_devices):
            info_dict = self.audio.get_device_info_by_index(i)
            logger.info([(key, value) for key, value in info_dict.items() if key in keys])

    def record(self):
        # start Recording
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk,
                        input_device_index=self.device)
        logger.info(f"Recording started for {self.duration} seconds")
        self.frames = []
        for i in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            self.frames.append(data)
        logger.info ("Recording Completed")
        # stop Recording
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        self.save()

    def save(self):
        waveFile = wave.open(self.path, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.sample_rate)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()
        logger.info(f"Recording saved to {self.path}")



def get_spectrogram(type='mel'):
    logger.info("Extracting spectrogram")
    y, sr = librosa.load(WAVE_OUTPUT_FILE, duration=DURATION)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logger.info("Spectrogram Extracted")
    format = '%+2.0f'
    if type == 'DB':
        ps = librosa.power_to_db(ps, ref=np.max)
        format = ''.join[format, 'DB']
        logger.info("Converted to DB scale")
    return ps, format

def display(spectrogram, format):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format=format)
    plt.tight_layout()
    st.pyplot(clear_figure=False)


def main():
    title = "Keyword Spotting UI"
    st.title(title)

    if st.button('Record'):
        sound = Sound()
        with st.spinner(f'Recording for {DURATION} seconds ....'):
            sound.record()
        st.success("Recording completed")

    if st.button('Play'):
        # sound.play()
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")

    if st.button('Classify'):
        #cnn = init_model()
        #with st.spinner("Classifying the chord"):
        #    chord = cnn.predict(WAVE_OUTPUT_FILE, False)
        st.success("Classification completed")
        CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')
        logits = np.random.randn(len(CLASSES))
        logits = logits * (logits > 0)
        logits_norm = logits / logits.sum()
        df = pd.DataFrame([logits_norm], columns=CLASSES)
        st.table(df)
        #st.write("### The recorded chord is **", chord + "**")
        #if chord == 'N/A':
        #    st.write("Please record sound first")
        #st.write("\n")

    # Add a placeholder
    if st.button('Display Spectrogram'):
        # type = st.radio("Scale of spectrogram:",
        #                 ('mel', 'DB'))
        if os.path.exists(WAVE_OUTPUT_FILE):
            spectrogram, format = get_spectrogram(type='mel')
            display(spectrogram, format)
        else:
            st.write("Please record sound first")


if __name__ == "__main__":
    main()
