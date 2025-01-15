import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import layers
import io
from PIL import Image
import tempfile
from collections import defaultdict

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
        
        original_img = cv2.resize(img_array, self.img_size)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(original_img)
        if bbox:
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            ax1.add_patch(rect)
        ax1.set_title('Detection')
        ax1.axis('off')
        
        ax2.imshow(cam, cmap='jet')
        ax2.set_title('Class Activation Map')
        ax2.axis('off')
        
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

class TripletMapper:
    def __init__(self, triplet_file, instrument_mapping_file):
        self.triplet_names = self._load_triplet_names(triplet_file)
        self.instrument_mappings = self._load_instrument_mappings(instrument_mapping_file)
    
    def _load_triplet_names(self, filepath):
        triplet_dict = {}
        with open(filepath, 'r') as f:
            for line in f:
                idx, triplet = line.strip().split(':')
                triplet_dict[int(idx)] = triplet
        return triplet_dict
    
    def _load_instrument_mappings(self, filepath):
        mappings = []
        with open(filepath, 'r') as f:
            next(f)  # Skip header
            for line in f:
                values = [int(x) for x in line.strip().split(',')]
                mappings.append(values)
        return mappings
    
    def get_triplet_name(self, triplet_idx):
        return self.triplet_names.get(triplet_idx, "Unknown Triplet")
    
    def get_instrument_idx(self, triplet_idx):
        if triplet_idx < len(self.instrument_mappings):
            return self.instrument_mappings[triplet_idx][1]
        return None

class VideoProcessor:
    def __init__(self, model):
        self.model = model
    
    def preprocess_frame(self, frame):
        processed_frame = cv2.resize(frame, (224, 224))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        processed_frame = processed_frame.astype(np.float32) / 255.0
        return processed_frame
    
    def process_video(self, video_file, confidence_threshold, triplet_mapper):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_file.read())
        temp_file.close()
        
        cap = cv2.VideoCapture(temp_file.name)
        unique_detections = defaultdict(list)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.preprocess_frame(frame)
            model_input = np.expand_dims(processed_frame, axis=0)
            predictions = self.model.predict(model_input, verbose=0)
            
            triplet_preds = predictions[1][0]
            
            for idx, confidence in enumerate(triplet_preds):
                if confidence > confidence_threshold:
                    triplet_name = triplet_mapper.get_triplet_name(idx)
                    instrument_idx = triplet_mapper.get_instrument_idx(idx)
                    detection_key = (triplet_name, instrument_idx)
                    
                    if confidence > max([conf for _, conf in unique_detections[detection_key]], default=0):
                        unique_detections[detection_key] = [(frame_count, confidence)]
            
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        return unique_detections

def main():
    st.title("Surgical Instrument Detection")
    
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
    
    # Initialize triplet mapper
    try:
        triplet_mapper = TripletMapper(r"C:\Users\satya\OneDrive\Documents\GitHub\BH\triplet.txt", 
                                     r"C:\Users\satya\OneDrive\Documents\GitHub\BH\maps.txt")
    except Exception as e:
        st.error(f"Error loading mapping files: {str(e)}")
        return

    # Input type selection
    input_type = st.radio("Select input type:", ["Image(For visualisation feature)", "Video"])
    
    instrument_classes = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', "other"]
    
    if input_type == "Image(For visualisation feature)":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.image(img_array, caption="Uploaded Image", use_container_width=True)
            
            confidence_threshold = st.slider(
                "Detection Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05
            )
            
            if st.button("Detect Instruments"):
                detector = CAMDetector(model, target_layer_name='re_lu_34')
                
                processed_img = tf.image.resize(img_array, (224, 224))
                processed_img = tf.cast(processed_img, tf.float32) / 255.0
                predictions = model.predict(np.expand_dims(processed_img, axis=0))
                
                st.subheader("Detected Instruments:")
                
                triplet_preds = predictions[1][0]
                detections_found = False
                
                for idx, confidence in enumerate(triplet_preds):
                    if confidence > confidence_threshold:
                        detections_found = True
                        triplet_name = triplet_mapper.get_triplet_name(idx)
                        instrument_idx = triplet_mapper.get_instrument_idx(idx)
                        
                        st.write(f"Triplet: {triplet_name} ({confidence:.2%} confidence)")
                        if instrument_idx is not None:
                            st.write(f"Instrument: {instrument_classes[instrument_idx]} ")
                        
                        fig = detector.visualize_detection(
                            img_array,
                            instrument_idx if instrument_idx is not None else 0,
                            instrument_classes[instrument_idx] if instrument_idx is not None else "Unknown",
                            confidence
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                
                if not detections_found:
                    st.info("No instruments detected above the confidence threshold.")
    
    else:  # Video processing
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    video_processor = VideoProcessor(model)
                    unique_detections = video_processor.process_video(uploaded_video, confidence_threshold, triplet_mapper)
                    
                    st.subheader("Unique Detections Throughout Video:")
                    
                    if unique_detections:
                        sorted_detections = sorted(
                            unique_detections.items(),
                            key=lambda x: x[1][0][1],
                            reverse=True
                        )
                        
                        for (triplet_name, instrument_idx), occurrences in sorted_detections:
                            frame_num, confidence = occurrences[0]
                            st.markdown(f"**Triplet Action:** {triplet_name}")
                            if instrument_idx is not None:
                                st.markdown(f"**Instrument:** {instrument_classes[instrument_idx]}")
                            st.markdown(f"**Highest Confidence:** {confidence:.2%} (Frame {frame_num})")
                            st.markdown("---")
                    else:
                        st.info("No detections above the confidence threshold.")

if __name__ == "__main__":
    main()