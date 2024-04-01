import cv2

# Distance from camera to object(face) measured in centimeters
Known_distance = 76.2

# Width of face in the real world or Object Plane measured in centimeters
Known_width = 14.3

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Defining the font
fonts = cv2.FONT_HERSHEY_COMPLEX

# Load the classifiers for detecting human faces, profile faces, and cats/dogs
face_detector = cv2.CascadeClassifier("/Users/rudrajoshi/PycharmProjects/pythonProject/.venv/haarcascade_frontalface_default.xml")
profile_face_detector = cv2.CascadeClassifier("/Users/rudrajoshi/PycharmProjects/pythonProject/.venv/haarcascade_profileface.xml")
catdog_detector = cv2.CascadeClassifier("/Users/rudrajoshi/PycharmProjects/pythonProject/.venv/haarcascade_frontalcatface_extended.xml")

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

# Function to detect human faces, including profile faces
def face_data(image):
    face_width = 0  # Initializing face width to zero

    # Converting color image to grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecting frontal faces in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    # If frontal faces are not detected, try detecting profile faces
    if len(faces) == 0:
        # Detecting profile faces in the image
        profile_faces = profile_face_detector.detectMultiScale(gray_image, 1.3, 5)
        # If profile faces are detected, get coordinates x, y , width and height
        if len(profile_faces) > 0:
            (x, y, h, w) = profile_faces[0]
            cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)
            face_width = w
    else:
        # If frontal faces are detected, get coordinates x, y , width and height
        (x, y, h, w) = faces[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)
        face_width = w

    # Return the face width in pixels
    return face_width

# Function to detect cats and dogs
def catdog_data(image):
    detection = None

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect cats and dogs
    catsdogs = catdog_detector.detectMultiScale(gray_image, 1.3, 5)

    # If any detection found, draw rectangle and return width
    if len(catsdogs) > 0:
        detection = ("CatDog", catsdogs[0])
        (x, y, h, w) = detection[1]
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)
        return w
    else:
        return 0

# Reading reference_image from directory
ref_image = cv2.imread("/Users/rudrajoshi/PycharmProjects/pythonProject/.venv/image.jpg")

# Find the face width (pixels) in the reference_image
ref_image_face_width = face_data(ref_image)

# Get the focal length by calling "Focal_Length_Finder"
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)

# Initialize the camera object
cap = cv2.VideoCapture(0)

# Main loop for video capture
while True:
    # Reading the frame from camera
    _, frame = cap.read()

    # Calling face_data function to find the width of face (pixels) in the frame
    face_width_in_frame = face_data(frame)

    # Calling catdog_data function to find the width of cat/dog (pixels) in the frame
    catdog_width_in_frame = catdog_data(frame)

    # Check if a human face is detected
    if face_width_in_frame != 0:
        # Finding the distance for human face
        Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
        detection_type = "Human Face"
    # Check if a cat or dog is detected
    elif catdog_width_in_frame != 0:
        # Finding the distance for cat/dog
        Distance = Distance_finder(Focal_length_found, Known_width, catdog_width_in_frame)
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
