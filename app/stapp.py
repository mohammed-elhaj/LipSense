# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')
# Setup the sidebar
with st.sidebar: 
    st.image('https://play.google.com/store/apps/details?id=digital.dong.lipreading&hl=en_US')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')
st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir('app/data/s1')
selected_video = st.selectbox('Choose video', options)
# Generate two columns 
col1, col2 = st.columns(2)
if options: 
    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        #file_path = os.path.join('..','data','s1', selected_video)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
        file_path = os.path.join(script_dir, 'data', 's1', selected_video)
        # Path to the output video
        output_file_path = os.path.join(script_dir, 'test_video.mp4')
        
        os.system(f'ffmpeg -i "{file_path}" -vcodec libx264 "{output_file_path}" -y')
        # Rendering inside of the app
        #video = open('app/test_video.mp4', 'rb') 
        #video_bytes = video.read() 
        #st.video(video_bytes)
        # Rendering inside of the app gpt 
        #video_path = os.path.join(script_dir,'test_video.mp4')
        with open(output_file_path, 'rb') as video:
            video_bytes = video.read()
        st.video(video_bytes)
    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()

        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)
        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
