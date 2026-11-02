import cv2
import numpy as np
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak the detected objects
def speak_detected_objects(objects):
    if objects:
        engine.say("I SEE " + ", ".join(objects))
        engine.runAndWait()

# Paths to the YOLOv3 files
weights_path = "/Users/naveenprathaps/PycharmProjects/Project_Blind/yolov3.weights"
config_path = "/Users/naveenprathaps/PycharmProjects/Project_Blind/yolov3.cfg"
names_path = "/Users/naveenprathaps/PycharmProjects/Project_Blind/coco.names"

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Get the names of all layers in the network
layer_names = net.getLayerNames()

# Safely access the output layers using flatten() to avoid IndexError
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load the class labels YOLO was trained on
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform a forward pass of YOLO to get the detections
    outputs = net.forward(output_layers)

    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    detected_objects = set()  # Use a set to avoid duplicate entries

    # Loop over each detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Get the class probabilities
            class_id = np.argmax(scores)  # Get the index of the class with the highest score
            confidence = scores[class_id]  # Get the confidence for this class
            if confidence > 0.5:  # Filter weak detections
                # Calculate the coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Add the detected class to the set of detected objects
                detected_objects.add(classes[class_id])

    # Perform non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels for detected objects
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding boxes

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Put the label and confidence score on the frame
            cv2.putText(frame, f"{label} {round(confidence, 2)}", (x, y + 30), font, 2, color, 2)

    # Speak out the detected objects
    speak_detected_objects(detected_objects)

    # Display the frame with the detections
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
