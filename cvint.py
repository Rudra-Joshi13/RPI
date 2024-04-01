import cv2

# Function to apply night vision effect
def apply_night_vision(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply night vision effect (in this case, simply equalize histogram)
    equalized = cv2.equalizeHist(gray)
    # Convert grayscale back to BGR (RGB) color space
    night_vision = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return night_vision

# URL of the video stream
url = "http://192.168.4.1:81/stream"  # Replace this with your URLq

# Open video stream
cap = cv2.VideoCapture(url)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is successfully captured
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Apply night vision effect
    night_vision_frame = apply_night_vision(frame)

    # Display the original and modified frames
    cv2.imshow('Original', frame)
    cv2.imshow('Night Vision', night_vision_frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()