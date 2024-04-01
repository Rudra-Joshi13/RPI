import cv2


Known_distance_human = 76.2
Known_distance_catdog = 50.0


Known_width_human = 14.3
Known_width_catdog = 10.0


GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

fonts = cv2.FONT_HERSHEY_COMPLEX


face_detector = cv2.CascadeClassifier("/Users/rudrajoshi/PycharmProjects/pythonProject/.venv/haarcascade_frontalface_default.xml")
catdog_detector = cv2.CascadeClassifier("/Users/rudrajoshi/PycharmProjects/pythonProject/.venv/haarcascade_frontalcatface.xml")

# Focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    # Finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# Distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    # Return the distance
    return distance

# Function to detect human faces
def face_data(image):
    face_width = 0  # Initializing face width to zero

    # Convert color image to grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecting human faces in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    # If human faces are detected, get coordinates x, y , width and height
    if len(faces) > 0:
        (x, y, h, w) = faces[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)
        face_width = w

    # Return the face width in pixels
    return face_width

# Function to detect cats/dogs
def catdog_data(image):
    catdog_width = 0  # Initializing cat/dog width to zero

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecting cat/dog faces in the image
    catdogs = catdog_detector.detectMultiScale(gray_image, 1.3, 5)

    # If any detection found, get coordinates x, y , width and height
    if len(catdogs) > 0:
        (x, y, h, w) = catdogs[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), RED, 2)
        catdog_width = w

    # Return the cat/dog width in pixels
    return catdog_width

# Reading reference human face image from directory
ref_image_human = cv2.imread("/Users/rudrajoshi/PycharmProjects/pythonProject/.venv/image.jpg")

# Reading reference cat/dog face image from directory
ref_image_catdog = cv2.imread("/Users/rudrajoshi/PycharmProjects/pythonProject/.venv/image.jpg")

# Find the human face width (pixels) in the reference image
ref_image_face_width_human = face_data(ref_image_human)

# Find the cat/dog face width (pixels) in the reference image
ref_image_face_width_catdog = catdog_data(ref_image_catdog)

# Get the focal length for human faces by calling "Focal_Length_Finder"
Focal_length_found_human = Focal_Length_Finder(Known_distance_human, Known_width_human, ref_image_face_width_human)

# Get the focal length for cat/dog faces by calling "Focal_Length_Finder"
Focal_length_found_catdog = Focal_Length_Finder(Known_distance_catdog, Known_width_catdog, ref_image_face_width_catdog)

# Initialize the camera object
cap = cv2.VideoCapture(0)

# Main loop for video capture
while True:
    # Reading the frame from camera
    _, frame = cap.read()

    # Calling face_data function to find the width of human face (pixels) in the frame
    face_width_in_frame = face_data(frame)

    # Calling catdog_data function to find the width of cat/dog face (pixels) in the frame
    catdog_width_in_frame = catdog_data(frame)

    # Check if a human face is detected
    if face_width_in_frame != 0:
        # Finding the distance for human face using human face focal length
        Distance = Distance_finder(Focal_length_found_human, Known_width_human, face_width_in_frame)
        detection_type = "Human Face"
    # Check if a cat/dog face is detected
    elif catdog_width_in_frame != 0:
        # Finding the distance for cat/dog face using cat/dog face focal length
        Distance = Distance_finder(Focal_length_found_catdog, Known_width_catdog, catdog_width_in_frame)
        detection_type = "Cat or Dog"
    else:
        # If no detection, set distance to 0 and detection type to None
        Distance = 0
        detection_type = "None"

    # Draw line as background of text
    cv2.line(frame, (30, 30), (230, 30), RED, 32)
    cv2.line(frame, (30, 30), (230, 30), BLACK, 28)

    # Drawing Text on the screen
    cv2.putText(frame, f"{detection_type}: {round(Distance, 2)} CM", (30, 35), fonts, 0.6, GREEN, 2)

    # Show the frame on the screen
    cv2.imshow("frame", frame)

    # Quit the program if 'q' is pressed on keyboard
    if cv2.waitKey(1) == ord("q"):
        break

# Closing the camera
cap.release()

# Closing the windows that are opened
cv2.destroyAllWindows()
