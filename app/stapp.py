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
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('تم تطوير هذا التطبيق أصلاً من نموذج التعلم العميق LipNet.')

st.title('تطبيق LipNet كامل الوظائف')

# Generating a list of options or videos
options = os.listdir('data/s1')
selected_video = st.selectbox('اختر فيديو', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:

    # Rendering the video
    with col1:
        st.info('الفيديو أدناه يعرض الفيديو المحول بتنسيق mp4')

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
        file_path = os.path.join(script_dir, 'data', 's1', selected_video)
        
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video_path = os.path.join(script_dir, 'test_video.mp4')
        with open(video_path, 'rb') as video:
            video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('هذا كل ما يراه نموذج التعلم الآلي عند القيام بالتنبؤ')
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        st.info('هذه هي مخرجات نموذج التعلم الآلي كرموز')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('فك تشفير الرموز الخام إلى كلمات')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
