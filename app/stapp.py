# Import all of the dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import tempfile

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')

# Add video upload feature
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

# Generating a list of options or videos from the app data folder
data_options = os.listdir('app/data/s1')
selected_video = st.selectbox('Or choose a sample video', data_options)

# Use uploaded video if available
if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        uploaded_video_path = tmp_file.name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(tempfile.gettempdir(), 'converted_video.mpg')
    os.system(f'ffmpeg -i "{uploaded_video_path}" -vcodec mpeg2video "{video_path}" -y')
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    video_path = os.path.join(script_dir, 'data', 's1', selected_video)

# Generate two columns
col1, col2 = st.columns(2)

if video_path:

    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')

        # Path to the output video
        output_file_path = os.path.join(tempfile.gettempdir(), 'test_video.mp4')
        
        os.system(f'ffmpeg -i "{video_path}" -vcodec libx264 "{output_file_path}" -y')

        # Rendering inside of the app
        with open(output_file_path, 'rb') as video:
            video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video_data, annotations = load_data(tf.convert_to_tensor(video_path))
        gif_path = os.path.join(tempfile.gettempdir(), 'animation.gif')
        imageio.mimsave(gif_path, video_data[:75], fps=10)
        st.image(gif_path, width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video_data[:75], axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
        # Adding model interpretation details
        st.info('Model confidence scores and other details')
        # Assuming yhat contains logits, we can apply a softmax to get probabilities
        confidence_scores = tf.nn.softmax(yhat[0]).numpy()
        
        # Display top-N predictions
        top_n = 5
        top_n_indices = confidence_scores.argsort()[-top_n:][::-1]
        for i in top_n_indices:
            char = num_to_char([i])[0].numpy().decode('utf-8')
            score = confidence_scores[i]
            st.write(f"Token {i}: {char} with confidence {score:.2%}")

# Enhancements for user experience and performance
st.info('Enhancements for better performance and user experience')
st.markdown("""
- Upload your own videos for lip reading.
- See real-time predictions and confidence scores.
- Improved UI/UX for a seamless experience.
""")
