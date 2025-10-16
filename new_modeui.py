import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import joblib
import os
import pandas as pd
import soundfile as sf
import io

# Load the trained model and preprocessing objects
@st.cache_resource
def load_resources():
    model_path = "cnn_model.h5"
    encoder_path = "label_encoder.pkl"
    scaler_path = "feature_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path) or not os.path.exists(scaler_path):
        st.error("Required model files not found. Please train the model first.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    return model, encoder, scaler

def extract_features(audio_file):
    try:
        # Reset file pointer and read uploaded bytes
        audio_file.seek(0)
        audio_bytes = audio_file.read()

        # Determine extension from uploaded filename when available
        orig_name = getattr(audio_file, 'name', None)
        suffix = '.wav'
        if orig_name:
            ext = os.path.splitext(orig_name)[1]
            if ext:
                suffix = ext

        # Write bytes to a temporary file using the detected suffix.
        # This helps libraries (librosa / soundfile / audioread) pick the right decoder.
        import tempfile
        tmp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        # First try to load with librosa (uses audioread which can handle mp3/flac/ogg if backends are present).
        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=30)
        except Exception as e_librosa:
            # Fallback to soundfile (works for many WAV variants). Provide detailed info on failure.
            try:
                y, sr = sf.read(tmp_path)
                # If multi-channel, convert to mono
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
            except Exception as e_soundfile:
                    # If both librosa and soundfile failed, try to transcode via ffmpeg to a WAV and reload.
                    # Try system ffmpeg first
                    try:
                        import subprocess
                        transcode_path = tmp_path + '_conv.wav'
                        ffmpeg_cmd = ['ffmpeg', '-y', '-i', tmp_path, '-ar', '22050', '-ac', '1', transcode_path]
                        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        y, sr = librosa.load(transcode_path, sr=None, mono=True, duration=30)
                        try:
                            os.remove(transcode_path)
                        except Exception:
                            pass
                    except FileNotFoundError:
                        # ffmpeg executable not found on PATH
                        try:
                            # Try imageio-ffmpeg (python-only) as a last resort
                            import imageio_ffmpeg as _i4f
                            transcode_path = tmp_path + '_conv.wav'
                            ffmpeg_path = _i4f.get_ffmpeg_exe()
                            import subprocess
                            subprocess.run([ffmpeg_path, '-y', '-i', tmp_path, '-ar', '22050', '-ac', '1', transcode_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            y, sr = librosa.load(transcode_path, sr=None, mono=True, duration=30)
                            try:
                                os.remove(transcode_path)
                            except Exception:
                                pass
                        except ModuleNotFoundError:
                            # Clean up and raise informative error
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                            raise Exception("ffmpeg executable not found on PATH and python package 'imageio-ffmpeg' is not installed. "
                                            "Install ffmpeg (add to PATH) or run 'pip install imageio-ffmpeg' in the same Python environment used to run Streamlit.")
                        except Exception as e:
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                            raise Exception(f"Transcoding with imageio-ffmpeg failed: {e}")
                    except Exception as e:
                        # Some other error occurred when trying system ffmpeg
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                        raise Exception(f"Transcoding with system ffmpeg failed: {e}")

        # Clean up temp file now that audio is loaded
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        
        # Initialize feature dictionary
        features = {}

        # Basic audio properties
        features['length'] = len(y)
        features['sample_rate'] = sr

        # Extract features
        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)
        features['chroma_stft_std'] = np.std(chroma_stft)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        
        # Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_var'] = np.var(spectral_centroid)
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # mfcc may have fewer than 20 coefficients depending on n_mfcc; guard against that
        for i in range(min(20, mfcc.shape[0])):
            features[f'mfcc{i+1}'] = np.mean(mfcc[i])
        for i in range(mfcc.shape[0], 20):
            # pad missing mfcc features with zeros
            features[f'mfcc{i+1}'] = 0.0
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)
        
        # Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)
        
        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
        features['zero_crossing_rate_var'] = np.var(zero_crossing_rate)
        
        # Harmony and Perceptr
        harmony, perceptr = librosa.effects.hpss(y)
        features['harmony_mean'] = np.mean(harmony)
        features['harmony_var'] = np.var(harmony)
        features['perceptr_mean'] = np.mean(perceptr)
        features['perceptr_var'] = np.var(perceptr)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # MFCCs (1-20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(min(20, mfccs.shape[0])):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_var'] = np.var(mfccs[i])
        for i in range(mfccs.shape[0], 20):
            features[f'mfcc{i+1}_mean'] = 0.0
            features[f'mfcc{i+1}_var'] = 0.0
        
        # Convert to DataFrame with single row
        features_df = pd.DataFrame([features])
        
        # Ensure columns are in the same order as training data
        columns_order = feature_scaler.feature_names_in_
        features_df = features_df[columns_order]
        
        return features_df
        
    except Exception as e:
        import traceback
        st.error(f"Error processing audio file: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

# Load model and preprocessing objects
model, label_encoder, feature_scaler = load_resources()

# --- Streamlit UI Layout ---
st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file to get its genre predicted.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Processing audio for prediction...")
    
    # Extract features
    features = extract_features(uploaded_file)
    
    if features is not None:
        # Scale features
        features_scaled = feature_scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        predicted_label_index = np.argmax(prediction[0])
        predicted_genre = label_encoder.inverse_transform([predicted_label_index])[0]
        
        st.success(f"The predicted genre is: **{predicted_genre.upper()}**")
        
        # Show confidence scores
        st.subheader("Model Confidence")
        confidence_df = pd.DataFrame(prediction[0], 
                                   index=label_encoder.classes_,
                                   columns=['Confidence'])
        st.bar_chart(confidence_df)