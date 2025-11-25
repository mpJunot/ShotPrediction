"""
Streamlit Web Application for Basketball Trajectory Analyzer
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
from PIL import Image
import time

# Add parent directory to path to import basketball_analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from basketball_analyzer import BasketballTrajectoryAnalyzer
from basketball_analyzer.config import *

# Page configuration
st.set_page_config(
    page_title="Basketball Trajectory Analyzer",
    page_icon=":basketball:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS from external file"""
    css_path = os.path.join(script_dir, 'styles.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. Using default styles.")

# Load CSS
load_css()

def main():
    st.markdown('<h1 class="main-header">Basketball Trajectory Analyzer</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Model selection section
        st.markdown("### Model Settings")
        default_model_path = os.path.join(project_root, 'models', 'shot.pt')
        model_path = st.text_input("Model Path", value=default_model_path, help="Path to the YOLO model file")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Analysis parameters
        st.markdown("### Analysis Parameters")
        fps = st.slider(
            "FPS (Frame Rate)",
            min_value=15,
            max_value=60,
            value=DEFAULT_FPS,
            help="Video frame rate for analysis"
        )
        px_per_meter = st.slider(
            "Pixels per Meter",
            min_value=50,
            max_value=300,
            value=DEFAULT_PX_PER_METER,
            help="Calibration factor: pixels per meter in the video"
        )
        min_speed_threshold = st.slider(
            "Min Speed Threshold (m/s)",
            min_value=0.1,
            max_value=5.0,
            value=MIN_SPEED_THRESHOLD,
            step=0.1,
            help="Minimum ball speed to trigger trajectory prediction"
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Visualization options
        st.markdown("### Visualization Options")

        # Initialize visualization options in session state
        if 'viz_options' not in st.session_state:
            st.session_state.viz_options = {
                'show_ball_trail': True,
                'show_trajectory': True,
                'show_detection_boxes': True,
                'show_probability': True
            }

        # Update options from checkboxes
        show_ball_trail = st.checkbox(
            "Show Ball Trail",
            value=st.session_state.viz_options['show_ball_trail'],
            help="Display ball position history",
            key='viz_ball_trail'
        )
        show_trajectory = st.checkbox(
            "Show Predicted Trajectory",
            value=st.session_state.viz_options['show_trajectory'],
            help="Display predicted trajectory path",
            key='viz_trajectory'
        )
        show_detection_boxes = st.checkbox(
            "Show Detection Boxes",
            value=st.session_state.viz_options['show_detection_boxes'],
            help="Display YOLO detection bounding boxes",
            key='viz_detection_boxes'
        )
        show_probability = st.checkbox(
            "Show Shot Probability",
            value=st.session_state.viz_options['show_probability'],
            help="Display shot success probability",
            key='viz_probability'
        )

        # Update session state
        st.session_state.viz_options = {
            'show_ball_trail': show_ball_trail,
            'show_trajectory': show_trajectory,
            'show_detection_boxes': show_detection_boxes,
            'show_probability': show_probability
        }

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Control buttons
        st.markdown("### Controls")
        col_reset, col_clear = st.columns(2)
        with col_reset:
            if st.button("Reset Analysis", width='stretch'):
                analyzer_key = st.session_state.get('current_analyzer_key')
                if analyzer_key and analyzer_key in st.session_state:
                    st.session_state[analyzer_key].reset_basket_position()
                    st.success("Basket position reset!")
                else:
                    st.info("No active analysis to reset")
        with col_clear:
            if st.button("Clear Stats", width='stretch'):
                if 'stats_history' in st.session_state:
                    st.session_state.stats_history = []
                    st.success("Statistics cleared!")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Live Analysis", "Statistics", "About"])

    with tab1:
        # Video input section
        st.header("Video Input")

        col_input1, col_input2 = st.columns([3, 1])

        with col_input1:
            input_option = st.radio(
                "Select Input Source",
                ["Upload Video", "Webcam", "Sample Video"],
                horizontal=True
            )

        video_source = None

        if input_option == "Upload Video":
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for analysis"
            )
            if uploaded_file is not None:
                # Save uploaded file temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_source = tfile.name

        elif input_option == "Webcam":
            if st.button("Start Webcam", width='stretch'):
                video_source = 0
                st.info("Webcam input selected. Processing will start automatically.")

        elif input_option == "Sample Video":
            sample_path = os.path.join(project_root, 'assets', 'basket.mp4')
            if os.path.exists(sample_path):
                if st.button("Load Sample Video", width='stretch'):
                    video_source = sample_path
            else:
                st.error("Sample video not found!")

        st.markdown("---")

        # Video processing section
        if video_source is not None:
            # Create two columns: video on left, metrics on right
            col_video, col_metrics = st.columns([2, 1])

            with col_video:
                st.subheader("Video Analysis")
                process_video(
                    video_source, model_path, fps, px_per_meter,
                    min_speed_threshold, show_ball_trail, show_trajectory,
                    show_detection_boxes, show_probability
                )

            with col_metrics:
                st.subheader("Real-time Metrics")
                display_realtime_metrics()
        else:
            # Show placeholder when no video is selected
            st.info("Please select a video source to begin analysis.")
            st.markdown("""
            ### Instructions:
            1. **Upload Video**: Select a video file from your computer
            2. **Webcam**: Use your webcam for live analysis
            3. **Sample Video**: Use the provided sample video

            Once a source is selected, the analysis will begin automatically.
            """)

    with tab2:
        st.header("Analysis Statistics")

        if 'stats_history' not in st.session_state or len(st.session_state.stats_history) == 0:
            st.info("Start video analysis to see statistics")
            st.markdown("""
            ### What you'll see here:
            - **Shot Probability Over Time**: Graph showing probability changes
            - **Ball Speed Over Time**: Graph showing speed variations
            - **Summary Statistics**: Average and maximum values
            """)
        else:
            display_statistics()

    with tab3:
        display_about_page()

def process_video(video_source, model_path, fps, px_per_meter, min_speed_threshold,
                 show_ball_trail, show_trajectory, show_detection_boxes, show_probability):
    """Process video with basketball analyzer"""

    try:
        # Initialize analyzer
        analyzer_key = f"analyzer_{model_path}_{fps}_{px_per_meter}"
        if analyzer_key not in st.session_state or st.session_state.get('model_path') != model_path:
            with st.spinner("Loading model..."):
                st.session_state[analyzer_key] = BasketballTrajectoryAnalyzer(
                    model_path=model_path,
                    fps=fps,
                    px_per_meter=px_per_meter,
                    show_ball_trail=show_ball_trail,
                    show_trajectory=show_trajectory,
                    show_detection_boxes=show_detection_boxes,
                    show_probability=show_probability
                )
                st.session_state.model_path = model_path
                st.session_state.current_analyzer_key = analyzer_key
                st.success("Model loaded successfully!")

        analyzer = st.session_state[st.session_state.get('current_analyzer_key', analyzer_key)]

        # Update visualization options if they changed
        analyzer.set_visualization_options(
            show_ball_trail=show_ball_trail,
            show_trajectory=show_trajectory,
            show_detection_boxes=show_detection_boxes,
            show_probability=show_probability
        )

        # Update calibration if changed
        if analyzer.px_per_meter != px_per_meter or analyzer.fps != fps:
            analyzer.set_calibration(px_per_meter)
            analyzer.fps = fps
            analyzer.trajectory_predictor.fps = fps

        # Open video
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            st.error("Failed to open video source")
            return

        # Video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        st.info(f"Video Info: {total_frames} frames, {video_fps:.1f} FPS")

        # Create placeholders
        frame_placeholder = st.empty()
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # Initialize session state for statistics
        if 'stats_history' not in st.session_state:
            st.session_state.stats_history = []

        frame_count = 0
        shot_detections = []

        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                st.info("End of video reached")
                break

            # Process frame (visualization options are already applied in analyzer)
            output, probability = analyzer.process_frame(frame)

            # Convert BGR to RGB for Streamlit
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            # Display frame
            frame_placeholder.image(output_rgb, channels="RGB", width='stretch')

            # Update progress
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_placeholder.progress(progress)

            # Collect statistics
            stats = analyzer.get_statistics()
            stats['frame'] = frame_count
            stats['probability'] = probability
            st.session_state.stats_history.append(stats)

            # Record shot detections
            if probability > 0.5:
                shot_detections.append({
                    'frame': frame_count,
                    'probability': probability,
                    'timestamp': frame_count / video_fps
                })

            # Display real-time metrics (will be shown in sidebar via display_realtime_metrics)
            # Store current stats for display
            st.session_state.current_stats = {
                'frame': frame_count,
                'total_frames': total_frames,
                'probability': probability,
                'speed': stats['current_speed'],
                'basket_detected': stats['basket_position'] is not None,
                'ball_tracked': stats['ball_positions_count'] > 0
            }

            frame_count += 1

            # Add small delay to control playback speed
            time.sleep(0.03)  # ~30 FPS

            # Break if stop button pressed (would need session state management)
            if frame_count % 10 == 0:  # Check every 10 frames
                if st.session_state.get('stop_processing', False):
                    break

        cap.release()

        # Display final results
        if shot_detections:
            st.success(f"Analysis complete! Detected {len(shot_detections)} potential shots")

            # Show shot summary
            st.subheader("Shot Summary")
            for i, shot in enumerate(shot_detections):
                st.write(f"Shot {i+1}: Frame {shot['frame']}, "
                        f"Time {shot['timestamp']:.1f}s, "
                        f"Probability {shot['probability']*100:.1f}%")
        else:
            st.info("Analysis complete! No shots detected")

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

def display_realtime_metrics():
    """Display real-time metrics in the sidebar"""
    if 'current_stats' not in st.session_state:
        st.info("Waiting for video analysis to start...")
        return

    stats = st.session_state.current_stats

    # Frame progress
    st.markdown("### Progress")
    progress = stats['frame'] / stats['total_frames'] if stats['total_frames'] > 0 else 0
    st.progress(progress)
    st.caption(f"Frame {stats['frame']} / {stats['total_frames']}")

    st.markdown("---")

    # Key metrics
    st.markdown("### Current Metrics")

    display_probability = stats['probability']
    if 'stats_history' in st.session_state and len(st.session_state.stats_history) > 0:
        recent_probs = [s['probability'] for s in st.session_state.stats_history[-30:]]
        if recent_probs:
            max_recent_prob = max(recent_probs)
            if stats['probability'] < 0.1 and max_recent_prob > 0.1:
                display_probability = max_recent_prob

    # Shot probability
    prob_color = "status-success" if display_probability > 0.6 else "status-warning" if display_probability > 0.3 else "status-error"
    prob_label = "Shot Probability" if stats['probability'] > 0 else "Peak Probability (Recent)"
    st.markdown(f"""
    <div class="metric-card">
        <h4>{prob_label}</h4>
        <h2 class="{prob_color}">{display_probability*100:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    # Ball speed
    st.markdown(f"""
    <div class="metric-card">
        <h4>Ball Speed</h4>
        <h2>{stats['speed']:.1f} m/s</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Detection status
    st.markdown("### Detection Status")
    if stats['basket_detected']:
        st.success("✓ Basket detected")
    else:
        st.warning("✗ Basket not detected")

    if stats['ball_tracked']:
        st.success("✓ Ball tracked")
    else:
        st.warning("✗ Ball not tracked")

def display_statistics():
    """Display analysis statistics and charts"""

    stats_history = st.session_state.stats_history

    if not stats_history:
        st.warning("No statistics available")
        return

    # Extract data for charts
    frames = [s['frame'] for s in stats_history]
    probabilities = [s['probability'] for s in stats_history]
    speeds = [s['current_speed'] for s in stats_history]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Shot Probability Over Time")
        chart_data = {
            'Frame': frames,
            'Probability': probabilities
        }
        st.line_chart(chart_data, x='Frame', y='Probability')

    with col2:
        st.subheader("Ball Speed Over Time")
        chart_data = {
            'Frame': frames,
            'Speed (m/s)': speeds
        }
        st.line_chart(chart_data, x='Frame', y='Speed (m/s)')

    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_prob = np.mean(probabilities) if probabilities else 0
        st.metric("Average Probability", f"{avg_prob*100:.1f}%")

    with col2:
        max_prob = max(probabilities) if probabilities else 0
        st.metric("Max Probability", f"{max_prob*100:.1f}%")

    with col3:
        avg_speed = np.mean(speeds) if speeds else 0
        st.metric("Average Speed", f"{avg_speed:.1f} m/s")

    with col4:
        max_speed = max(speeds) if speeds else 0
        st.metric("Max Speed", f"{max_speed:.1f} m/s")

def display_about_page():
    """Display about page with project information"""

    st.markdown("""
    ## About Basketball Trajectory Analyzer

    This application uses computer vision and physics-based modeling to analyze basketball shots in real-time.

    ### Features
    - Real-time basketball, rim, and player detection using YOLOv8
    - Physics-based trajectory prediction
    - Shot probability calculation
    - Real-time metrics and statistics
    - Support for video files and webcam input

    ### How it Works
    1. **Detection**: YOLOv8 model detects basketball, rim, and players
    2. **Tracking**: Ball position is tracked over time
    3. **Physics**: Projectile motion equations predict trajectory
    4. **Analysis**: Shot probability calculated based on trajectory accuracy

    ### Model Requirements
    The YOLO model should detect:
    - Class 0: Basketball
    - Class 1: Player
    - Class 2: Basketball rim

    ### Configuration
    - **FPS**: Frame rate for analysis
    - **Pixels per Meter**: Scale conversion for physics calculations
    - **Min Speed Threshold**: Minimum ball speed for shot detection

    ### Technical Details
    - Built with Streamlit for web interface
    - Uses OpenCV for video processing
    - Physics calculations based on projectile motion
    - Real-time visualization with customizable options

    ### Support
    For issues or questions, please check the project documentation or create an issue on GitHub.
    """)

    # System information
    st.subheader("System Information")
    col1, col2 = st.columns(2)

    with col1:
        st.info("**Framework**: Streamlit")
        st.info("**Computer Vision**: OpenCV")
        st.info("**Object Detection**: YOLOv8")

    with col2:
        st.info("**Physics Engine**: Custom projectile motion")
        st.info("**Visualization**: Real-time overlay")
        st.info("**Input Support**: Video files, Webcam")

if __name__ == "__main__":
    main()
