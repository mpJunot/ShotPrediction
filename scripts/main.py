"""
Main application for Basketball Trajectory Analyzer
"""

import cv2
import os
import sys

# Add parent directory to path to import basketball_analyzer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basketball_analyzer import BasketballTrajectoryAnalyzer


def main():
    """Main application function"""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Initialize analyzer
    model_path = os.path.join(project_root, 'models', 'shot.pt')
    analyzer = BasketballTrajectoryAnalyzer(model_path)

    # Open video capture
    video_path = os.path.join(project_root, 'assets', 'basket.mp4')
    cap = cv2.VideoCapture(video_path)

    # Playback control
    paused = False
    playback_speed = 1.0

    print("Basketball Trajectory Analyzer")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  q - Quit")
    print("  + - Increase speed")
    print("  - - Decrease speed")
    print("  r - Reset basket position")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break

            # Process frame with analyzer - returns 2 values
            output, probability = analyzer.process_frame(frame)

            # Optional: Display shot probability in console
            if probability > 0:
                print(f"Shot probability: {probability*100:.1f}%")

        else:
            # When paused, use the last processed frame
            if 'output' in locals():
                analyzer.visualizer.draw_pause_indicator(output)

        # Display the frame
        if 'output' in locals():
            cv2.imshow('Basketball Trajectory Analyzer', output)

        # Handle keyboard input
        wait_time = int(30 / playback_speed)
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('+') or key == ord('='):
            playback_speed = min(3.0, playback_speed + 0.25)
            print(f"Speed: {playback_speed:.2f}x")
        elif key == ord('-'):
            playback_speed = max(0.25, playback_speed - 0.25)
            print(f"Speed: {playback_speed:.2f}x")
        elif key == ord('r'):
            analyzer.reset_basket_position()
            print("Basket position reset")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
