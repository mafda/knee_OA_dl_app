import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img
    )

    return superimposed_img


icon = Image.open("app/img/mdc.png")
st.set_page_config(
    page_title="Severity Analysis of Arthrosis in the Knee",
    page_icon=icon,
)

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

model = tf.keras.models.load_model("./src/models/model_Xception_ft.hdf5")
target_size = (224, 224)

# Grad-CAM
grad_model = tf.keras.models.clone_model(model)
grad_model.set_weights(model.get_weights())
grad_model.layers[-1].activation = None
grad_model = tf.keras.models.Model(
    inputs=[grad_model.inputs],
    outputs=[
        grad_model.get_layer("global_average_pooling2d_1").input,
        grad_model.output,
    ],
)

# Sidebar
with st.sidebar:
    st.image(icon)
    st.subheader("Final Project - MDC013")
    st.caption("=== Fernanda Rodriguez ===")

    st.subheader(":arrow_up: Upload image")
    uploaded_file = st.file_uploader("Choose x-ray image")


# Body
st.header("Severity Analysis of Arthrosis in the Knee")

col1, col2 = st.columns(2)
y_pred = None

if uploaded_file is not None:
    with col1:
        st.subheader(":camera: Input")
        st.image(uploaded_file, use_column_width=True)

        img = tf.keras.preprocessing.image.load_img(
            uploaded_file, target_size=target_size
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        img_aux = img.copy()

        if st.button(
            ":arrows_counterclockwise: Predict Arthrosis in the Knee"
        ):
            img_array = np.expand_dims(img_aux, axis=0)
            img_array = np.float32(img_array)
            img_array = tf.keras.applications.xception.preprocess_input(
                img_array
            )

            with st.spinner("Wait for it..."):
                y_pred = model.predict(img_array)

            y_pred = 100 * y_pred[0]

            probability = np.amax(y_pred)
            number = np.where(y_pred == np.amax(y_pred))
            grade = str(class_names[np.amax(number)])

            st.subheader(":white_check_mark: Prediction")

            st.metric(
                label="Severity Grade:",
                value=f"{class_names[np.amax(number)]} - {probability:.2f}%",
            )

    if y_pred is not None:
        with col2:
            st.subheader(":mag: Explainability")
            heatmap = make_gradcam_heatmap(grad_model, img_array)
            image = save_and_display_gradcam(img, heatmap)
            st.image(image, use_column_width=True)

            st.subheader(":bar_chart: Analysis")

            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(class_names, y_pred, height=0.55, align="center")
            for i, (c, p) in enumerate(zip(class_names, y_pred)):
                ax.text(p + 2, i - 0.2, f"{p:.2f}%")
            ax.grid(axis="x")
            ax.set_xlim([0, 120])
            ax.set_xticks(range(0, 101, 20))
            fig.tight_layout()
            st.pyplot(fig)
