import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
from utils.streamlit.predict_calls import PredictAPI

# if "uploaded_image" not in st.session_state:
#     st.session_state.uploaded_image = None

if "predict" not in st.session_state:
    st.session_state.predict = PredictAPI()

if "text_data" not in st.session_state:
    st.session_state.text_data = ""


st.title("FastSAM Inference Tool! 😊")

temp_subheader = st.empty()
temp_uploader = st.empty()
temp_subheader.subheader("Upload an Image")
uploaded_image = temp_uploader.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    temp_uploader.empty()
    temp_subheader.empty()

    st.session_state.uploaded_image = Image.open(uploaded_image)
    img_byte_arr = BytesIO()
    st.session_state.uploaded_image.save(img_byte_arr, format='PNG')
    st.session_state.image_bytes = img_byte_arr.getvalue()

    st.sidebar.title("Mode Selection")
    mode = st.sidebar.radio("Select Mode", ["Everything", "Box", "Text", "Points"])

    if mode == "Everything":
        st.subheader("Everything Mode")
        uploaded_image = st.session_state.uploaded_image

        everything_container = st.empty()
        everything_container.image(uploaded_image, caption="Uploaded Image")    
    
        if st.button("Predict"):
            result = st.session_state.predict.predict_everything(st.session_state.image_bytes)
            if result != "Prediction Error":
                everything_container.image(result.content, caption="Result Image")
            else:
                st.write("Prediction Error")

    elif mode == "Text":

        st.subheader("Text Mode")
        uploaded_image = st.session_state.uploaded_image

        text_container = st.empty()
        text_container.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        text_input = st.text_input("Enter Text", "")

        if st.button("Add Text"):
            if text_input:
                st.session_state.text_data = text_input
                st.success(f"Text added: {text_input}")
                text_input = ""
            
        if len(st.session_state.text_data) > 0:
            if st.button("Predict"):
                result = st.session_state.predict.predict_text(st.session_state.image_bytes, 
                                                                st.session_state.text_data)
                if result != "Prediction Error":
                    text_container.image(result.content, caption="Result Image")
                else:
                    st.write("Prediction Error")

    elif mode == "Box":
        st.subheader("Box Mode")
        box_image = st.session_state.uploaded_image
        box_container = st.empty()

        with box_container.container():

            label_color = (
            st.sidebar.color_picker("Annotation color: ", "#F5F0F0") + "77"
            )
            mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "rect"
            canvas_result = st_canvas(
                fill_color=label_color,
                stroke_width=1,
                background_image=box_image,
                height=320,
                width=512,
                drawing_mode=mode,
                key="box"
            )
        if canvas_result.json_data is not None:
            df = pd.json_normalize(canvas_result.json_data["objects"])
            if len(df) > 0:
                df['x_top_left'] = df['left'] / 512
                df['y_top_left'] = df['top'] / 320
                df['x_bottom_right'] = (df['left'] + df['width'])/ 512
                df['y_bottom_right'] = (df['top'] + df['height']) / 320
                st.dataframe(df[["x_top_left", "y_top_left", "x_bottom_right", "y_bottom_right"]])                

        if st.button("Predict") and len(df)>0:
            zipped_columns = df[["x_top_left", 
                                    "y_top_left", 
                                    "x_bottom_right", 
                                    "y_bottom_right"]].values.tolist()
            result = st.session_state.predict.predict_box(st.session_state.image_bytes, 
                                                            zipped_columns)
            if result != "Prediction Error":
                box_container.image(result.content, caption="Result Image")
            else:
                st.write("Prediction Error")          

    elif mode == "Points":
        st.subheader("Point Mode")
        point_image = st.session_state.uploaded_image
        point_container = st.empty()

        with point_container.container():

            #09EA29 Green #EA1909 Red
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
            mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "point"

            label_type = st.sidebar.selectbox("Label Type", ["Foreground", "Background"])
            
            if label_type == "Foreground":
                label_color = "#09EA29"
            else:
                label_color = "#EA1909"

            canvas_result = st_canvas(
                fill_color=label_color,
                stroke_width=1,
                background_image=point_image,
                height=320,
                width=512,
                drawing_mode=mode,
                point_display_radius=point_display_radius,
                key="point"
            )
        if canvas_result.json_data is not None:
            df = pd.json_normalize(canvas_result.json_data["objects"])
            if len(df) > 0:
                df['top'] = np.maximum(df['top'], 0)
                df['left'] = np.maximum(df['left'], 0)
                df['label'] = np.where(df['fill'] == '#09EA29', 1, 0)
                df['x'] = df['left'] / 512
                df['y'] = df['top'] / 320

                st.dataframe(df[["x", "y", "label"]])

        if st.button("Predict") and len(df) > 0:
            points = df[["x", "y"]].values.tolist()
            pointlabels = df["label"].values.tolist()
            result = st.session_state.predict.predict_points(st.session_state.image_bytes, 
                                                                points, pointlabels)
            if result != "Prediction Error":
                point_container.image(result.content, caption="Result Image")
            else:
                st.write("Prediction Error")
