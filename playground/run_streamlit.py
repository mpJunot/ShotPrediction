"""
Simple script to run the Streamlit Basketball Analyzer app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""

    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])

    # Change to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    # Run the Streamlit app from playground directory
    streamlit_app_path = os.path.join('playground', 'streamlit_app.py')
    cmd = [sys.executable, "-m", "streamlit", "run", streamlit_app_path]

    print("Starting Basketball Trajectory Analyzer Web App...")
    print("The app will open in your default web browser")
    print("Press Ctrl+C to stop the app")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main()
