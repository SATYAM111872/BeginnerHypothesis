import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import layers
import io
from PIL import Image

# Custom ULSAM Layer definition
class ULSAMLayer(layers.Layer):
    def __init__(self, groups=8, **kwargs):
        super(ULSAMLayer, self).__init__(**kwargs)
        self.groups = groups
        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=1, strides=1, padding='same')
        self.max_pool = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')
        self.conv = layers.Conv2D(filters=80, kernel_size=1, strides=1, padding='same')

    def call(self, input_tensor):
        channels = input_tensor.shape[-1]
        group_size = channels // self.groups
        splits = tf.split(input_tensor, num_or_size_splits=self.groups, axis=-1)
        output_splits = []
        for split in splits:
            processed_split = self.process_split(split, group_size)
            output_splits.append(processed_split)
        return tf.concat(output_splits, axis=-1)

    def process_split(self, split, group_size):
        x = self.depthwise_conv(split)
        x = self.max_pool(x)
        x = self.conv(x)
        return x

class CAMDetector:
    def __init__(self, model, target_layer_name, img_size=(224, 224)):
        self.model = model
        self.target_layer = model.get_layer(target_layer_name)
        self.img_size = img_size
        self.cam_model = tf.keras.Model(
            inputs=model.input,
            outputs=[self.target_layer.output, model.get_layer('instrument_output').output]
        )
    
    def preprocess_image(self, img_array):
        """Modified to accept numpy array instead of path"""
        img_array = cv2.resize(img_array, self.img_size)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return tf.convert_to_tensor(img_array)
        
    def generate_cam(self, img, class_idx):
        with tf.GradientTape() as tape:
            tape.watch(img)
            conv_output, predictions = self.cam_model(img)
            loss = predictions[0, class_idx]
            
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        cam = tf.maximum(cam, 0)
        if tf.reduce_max(cam) > 0:
            cam = cam / tf.reduce_max(cam)
        cam = tf.squeeze(cam)
        return cam.numpy()
    
    def get_bounding_box(self, cam, threshold=0.5):
        cam_resized = cv2.resize(cam, self.img_size)
        binary = (cam_resized > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)
    
    def visualize_detection(self, img_array, class_idx, instrument_name, confidence):
        img_tensor = self.preprocess_image(img_array)
        cam = self.generate_cam(img_tensor, class_idx)
        bbox = self.get_bounding_box(cam)
        
        # Resize image for display
        original_img = cv2.resize(img_array, self.img_size)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with bbox
        ax1.imshow(original_img)
        if bbox:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            ax1.add_patch(rect)
        ax1.set_title('Detection')
        ax1.axis('off')
        
        # CAM heatmap
        ax2.imshow(cam, cmap='jet')
        ax2.set_title('Class Activation Map')
        ax2.axis('off')
        
        # Overlay
        cam_resized = cv2.resize(cam, self.img_size)
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)
        ax3.imshow(overlay)
        if bbox:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            ax3.add_patch(rect)
        ax3.set_title(f'{instrument_name} ({confidence:.2f})')
        ax3.axis('off')
        
        plt.tight_layout()
        return fig

def main():
    st.title("Surgical Instrument Detection")
    st.write("Upload an image to detect surgical instruments in laparoscopic procedures")
    
    # Model loading
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(
            "model_final.h5",
            custom_objects={'ULSAMLayer': ULSAMLayer}
        )
    
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.image(img_array, caption="Uploaded Image", use_column_width=True)
        
        # Add confidence threshold slider
        confidence_threshold = st.slider(
            "Detection Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05
        )
        
        # Process button
        if st.button("Detect Instruments"):
            detector = CAMDetector(model, target_layer_name='re_lu_34')
            instrument_classes = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator',"other"]
            
            # Get model predictions
            processed_img = tf.image.resize(img_array, (224, 224))
            processed_img = tf.cast(processed_img, tf.float32) / 255.0
            predictions = model.predict(np.expand_dims(processed_img, axis=0))
            
            # Display predictions
            st.subheader("Detected Instruments:")
            
            # Process detections
            triplet_preds = predictions[1][0]
            detections_found = False
            
            for idx, confidence in enumerate(triplet_preds):
                if confidence > confidence_threshold:
                    detections_found = True
                    st.write(f"- {instrument_classes[idx]}: {confidence:.2%} confidence")
                    fig = detector.visualize_detection(
                        img_array,
                        idx,
                        instrument_classes[idx],
                        confidence
                    )
                    st.pyplot(fig)
                    plt.close(fig)
            
            if not detections_found:
                st.info("No instruments detected above the confidence threshold.")

if __name__ == "__main__":
    main()