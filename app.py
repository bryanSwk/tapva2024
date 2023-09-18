import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
from utils.streamlit.predict_calls import PredictAPI

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "predict" not in st.session_state:
    st.session_state.predict = PredictAPI()

st.sidebar.title("Mode Selection")
mode = st.sidebar.radio("Select Mode", ["Upload Image", "Everything", "Box", "Text", "Points"])

if "text_data" not in st.session_state:
    st.session_state.text_data = ''

st.title("FastSAM Inference Tool! 😊")

st.session_state.points = []
st.session_state.pointlabels = []

def draw_bounding_box(image, coordinates):
    draw = ImageDraw.Draw(image)
    x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(int, coordinates)
    draw.rectangle([(x_top_left, y_top_left), (x_bottom_right, y_bottom_right)], outline="red", width=3)
    
    return image

def draw_points(image, points, labels, radius=3):
    draw = ImageDraw.Draw(image) 
    for i, point in enumerate(points):
        x, y = point
        label = labels[i]
        color = "red" if label == 0 else "green"
        x0, y0 = x - radius, y - radius
        x1, y1 = x + radius, y + radius
        draw.ellipse([(x0, y0), (x1, y1)], outline=color, fill=color)
    
    return image

if mode == "Upload Image":
    st.subheader("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.session_state.uploaded_image = Image.open(uploaded_image)
        img_byte_arr = BytesIO()
        st.session_state.uploaded_image.save(img_byte_arr, format='PNG')
        st.session_state.image_bytes = img_byte_arr.getvalue()
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("Please select a mode for inference.")

elif mode == "Everything":
    st.subheader("Everything Mode")
    uploaded_image = st.session_state.uploaded_image

    if uploaded_image is not None:
        image_container = st.empty()
        image_container.image(uploaded_image, caption="Uploaded Image")    
    
    if st.button("Predict"):
        result = st.session_state.predict.predict_everything(st.session_state.image_bytes)
        image_container.image(result.content, caption="Result Image")

elif mode == "Text":

    st.subheader("Text Mode")
    uploaded_image = st.session_state.uploaded_image

    if uploaded_image is not None:
        image_container = st.empty()
        image_container.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        text_input = st.text_input("Enter Text", "")

        if st.button("Add Text"):
            if text_input:
                st.session_state.text_data = text_input
                st.success(f"Text added: {text_input}")
                text_input = ""
        
        if len(st.session_state.text_data) > 0:
            if st.button("Predict"):
                result = st.session_state.predict.predict_text(st.session_state.image_bytes, st.session_state.text_data)
                image_container.image(result.content, caption="Result Image", use_column_width=True)


elif mode == "Box":
    st.subheader("Box Mode")
    uploaded_image = st.session_state.uploaded_image
    if uploaded_image is not None:
        image_container = st.empty()
        image_container.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.write("Enter 4 comma-separated integers to draw a bounding box.")

        bbox_input = st.text_input("Bounding Box (x_top_left, y_top_left, x_bottom_right, y_bottom_right)")

        if len(bbox_input.split(",")) == 4:
            coordinates = [int(coord.strip()) for coord in bbox_input.split(",")]

            width, height = uploaded_image.size

            coordinates[0] = coordinates[0] / width
            coordinates[1] = coordinates[1] / height
            coordinates[2] = coordinates[2] / width
            coordinates[3] = coordinates[3] / height
            
            img_with_bbox = draw_bounding_box(uploaded_image, coordinates)
            image_container.image(img_with_bbox, caption="Image with Bounding Box", use_column_width=True)

        if st.button("Predict"):
            result = st.session_state.predict.predict_box(st.session_state.image_bytes, [coordinates])
            image_container.image(result.content, caption="Result Image")


elif mode == "Points":
    st.subheader("Point Mode")
    uploaded_image = st.session_state.uploaded_image

    if uploaded_image is not None:
        image_container = st.empty()
        image_container.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        width, height = uploaded_image.size

        point_box = st.text_input("Point (x, y)")
        point_label = st.text_input("Label (0 or 1)")

    if len(point_box.split(",")) == 2 and len(point_label) == 1:
        try:
            x, y = map(int, point_box.split(","))
            label = int(point_label)
            if label in [0, 1]:
                st.session_state.points.append([x, y])
                st.session_state.pointlabels.append(label)
                img_with_points = draw_points(uploaded_image, st.session_state.points, st.session_state.pointlabels)
                image_container.image(img_with_points, caption="Image with Points", use_column_width=True)
                point_box = ""
                point_label = ""
            else:
                st.warning("Label must be 0 or 1.")
        except ValueError:
            st.warning("Invalid point coordinates or label. Please use the format 'x,y' for coordinates and enter a valid label.")

    if st.button("Predict"):
        points = [[point[0]/width, point[1]/height] for point in st.session_state.points]
        result = st.session_state.predict.predict_points(st.session_state.image_bytes, points, st.session_state.pointlabels)
        image_container.image(result.content, caption="Result Image")


