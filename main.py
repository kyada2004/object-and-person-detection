import cv2
import dlib
import numpy as np
import pyttsx3
import argparse

tts_engine = pyttsx3.init()

def load_yolo_model(cfg, weights):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    return net

def get_yolo_class_labels(names_file):
    with open(names_file, 'r') as f:
        class_labels = f.read().strip().split('\n')
    return class_labels

def calculate_distance(pixel_width, known_width, focal_length):
    return (known_width * focal_length) / pixel_width

def calculate_direction(center_x, frame_width):
    relative_x = (center_x - frame_width / 2) / (frame_width / 2)
    if relative_x < -0.33:
        return "left"
    elif relative_x > 0.33:
        return "right"
    else:
        return "center"
        
def detect_objects_yolo(frame, net, classes, focal_length, known_width, confidence_threshold=0.2):
    speak_text = ""  
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    detections = net.forward(output_layers)
    
    (h, w) = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, confidence_threshold)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            y_label = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, label, (x, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            distance = calculate_distance(w, known_width, focal_length)
            distance_text = f"Distance: {distance:.2f} units"
            cv2.putText(frame, distance_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            direction = calculate_direction(centerX, w)
            direction_text = f"Direction: {direction}"
            cv2.putText(frame, direction_text, (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            speak_text += f"Detected {classes[class_ids[i]]} at {distance:.2f} units to the {direction}. "

    return frame, speak_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    yoloCfg = "yolov3.cfg"
    yoloWeights = "yolov3.weights"
    yoloNames = "coco.names"

    model_mean = (78.4263377603, 87.7689143744, 114.895847746)

    ageList = ['(0-5)', '(6-11)', '(12-17)', '(18-20)', '(21-27)', '(27-35)', '(36-44)', '(45-53)', '(54-62)', '(63-71)', '(72-80)', '(81-89)', '(90-92)', '(93-98)', '(99-100)']
    genderList = ['Male', 'Female']
    classLabels = get_yolo_class_labels(yoloNames)

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    yoloNet = load_yolo_model(yoloCfg, yoloWeights)

    face_detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)

    KNOWN_FACE_WIDTH = 0.16  
    FOCAL_LENGTH = 500  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(img_gray)

        frame, object_speak_text = detect_objects_yolo(frame, yoloNet, classLabels, FOCAL_LENGTH, KNOWN_FACE_WIDTH)

        face_speak_text = ""
        if not faces:
            mssg = 'No face detected'
            cv2.putText(frame, f'{mssg}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for face in faces:
                x = face.left()
                y = face.top()
                x2 = face.right()
                y2 = face.bottom()

                x = max(0, x)
                y = max(0, y)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                face_img = frame[y:y2, x:x2]

                if face_img.size == 0:
                    continue

                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), model_mean, swapRB=False)

                genderNet.setInput(blob)
                gender_preds = genderNet.forward()
                gender = genderList[gender_preds[0].argmax()]

                ageNet.setInput(blob)
                age_preds = ageNet.forward()
                age = ageList[age_preds[0].argmax()]

                cv2.putText(frame, f'Gender: {gender}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                pixel_width = x2 - x
                distance = calculate_distance(pixel_width, KNOWN_FACE_WIDTH, FOCAL_LENGTH)
                distance_text = f"Distance: {distance:.2f} meters"
                cv2.putText(frame, distance_text, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                centerX = (x + x2) // 2
                direction = calculate_direction(centerX, frame.shape[1])
                direction_text = f"Direction: {direction}"
                cv2.putText(frame, direction_text, (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                face_speak_text += f"Detected face of a {gender} aged {age} at {distance:.2f} meters to the {direction}. "

        combined_speak_text = object_speak_text + face_speak_text
        if combined_speak_text:
            tts_engine.say(combined_speak_text)
            tts_engine.runAndWait()

        cv2.imshow("Object, Age, Gender, and Distance Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
