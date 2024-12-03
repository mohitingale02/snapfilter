from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Function to overlay the foreground image on the webcam feed
def overlay_foreground_on_camera(foreground_path):
    # Load the foreground image (with alpha channel if present)
    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)

    if foreground is None:
        print("Error: Unable to load the foreground image.")
        return

    # Check if the image has an alpha channel
    if foreground.shape[2] == 4:
        alpha_channel = foreground[:, :, 3] / 255.0
        rgb_foreground = foreground[:, :, :3]
    else:
        print("No transparency detected in the foreground image.")
        rgb_foreground = foreground
        alpha_channel = np.ones(rgb_foreground.shape[:2], dtype=np.float32)

    # Get the dimensions of the foreground image
    fg_h, fg_w, _ = rgb_foreground.shape

    # Start video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Rotate the frame to make it vertical
        frame = cv2.transpose(frame)  # Transpose the frame
        frame = cv2.flip(frame, flipCode=1)  # Flip to make it upright

        # Resize the camera frame to match the foreground dimensions
        frame = cv2.resize(frame, (fg_w, fg_h))

        # Overlay the foreground onto the camera feed
        for c in range(3):  # Apply to all color channels
            frame[:, :, c] = (
                alpha_channel * rgb_foreground[:, :, c]
                + (1 - alpha_channel) * frame[:, :, c]
            )

        # Encode the frame to JPEG format for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in a multipart response for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    foreground_path = "static/tattoo.png"  # Path to your foreground image
    return Response(overlay_foreground_on_camera(foreground_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)