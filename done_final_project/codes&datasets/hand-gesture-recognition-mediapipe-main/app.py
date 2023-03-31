import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import cv2 as cv2
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from model import PointHistoryClassifier

def get_arguments():
    interpreter = argparse.ArgumentParser()
    #adding a camera
    interpreter.add_argument("--device", type=int, default=0)
    #add width and height
    interpreter.add_argument("--width", help='cap width', type=int, default=960)
    interpreter.add_argument("--height", help='cap height', type=int, default=540)
    #add static image mode
    interpreter.add_argument('--use_static_image_mode', action='store_true')
    interpreter.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    interpreter.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    #add arguments
    argument = interpreter.parse_args()
    return argument
    #create a mask to cover

def main():
    # Argument parsing #
    config = get_arguments()#get the arguments
    #assign the arguments
    cap_input = config.device
    cap_width = config.width
    cap_height = config.height
    #assign the arguments
    use_static_image_mode = config.use_static_image_mode
    min_detection_confidence = config.min_detection_confidence
    min_tracking_confidence = config.min_tracking_confidence
    use_brect = True

    # Camera preparation #
    cap = cv2.VideoCapture(cap_input)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2, #number of hands
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    #load the model for the keypoint classifier
    keypoint_classifier = KeyPointClassifier()
    #load the model for the point history classifier
    point_history_classifier = PointHistoryClassifier()

    # Read labels #
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f: #encoding the labels
        keypoint_classifier_labels = csv.reader(f) #read the labels
        keypoint_classifier_labels = [ #assign the labels
            row[0] for row in keypoint_classifier_labels #assign the labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f: #encoding the labels
        point_history_classifier_labels = csv.reader(f) #read the labels
        point_history_classifier_labels = [ #assign the labels
            row[0] for row in point_history_classifier_labels #assign the labels
        ]


    # Coordinate history #
    historyLength = 16 #history length
    point_history = deque(maxlen=historyLength) #point history

    # Finger gesture history #
    finger_gesture_history = deque(maxlen=historyLength)#finger gesture history

    mode = 0
    #reduce brightness of the image
    def adjust_gamma(image, gamma=2.0): #reduce brightness of the image
        invGamma = 1 / gamma #inverse gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 #table for the gamma correction
                          for i in np.arange(0, 256)]).astype("uint8") #table for the gamma correction
        return cv2.LUT(image, table) #return the gamma corrected image

    while True:

        # Process Key (ESC: end) #
        key = cv2.waitKey(10) #wait for the key
        if key == 27:  #ESC
            break
        number, mode = modeSelect(key, mode) #select the mode

        # Camera capture #
        ret, image = cap.read()
        if not ret:
            break
        # image = add_gaussian_noise(image)  # Add noise to the image
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Create a mask to cover half of the screen
        mask = np.zeros_like(image)
        mask[:, :image.shape[1]//2] = 255  # Change this line to choose which half to block (left or right)
        # Apply the mask to the input image
        image = cv2.bitwise_and(image, mask)

        # apply gamma correction
        gamma = 1
        image = adjust_gamma(image, gamma)
        debug_image = copy.deepcopy(image)

        # Detection implementation #
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert the image to RGB
        image.flags.writeable = False # Make the image read-only
        results = hands.process(image) #process the image
        image.flags.writeable = True # Make the image read-write
        # Draw the hand landmarks on the debug_image
        if results.multi_hand_landmarks is not None: #if the results are not none
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, #zip the results
                                                  results.multi_handedness): #zip the results
                # # Accuracy calculation
                accuracy_percentage = int(
                    results.multi_handedness[0].classification[0].score * 100)

                # Draw accuracy percentage on image
                if accuracy_percentage > 80:
                    cv2.putText(debug_image,
                               "Accuracy: {}%".format(accuracy_percentage),
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 255, 0), 2, cv2.LINE_AA)
                    color = (0, 255, 0)
                elif accuracy_percentage > 60:
                    cv2.putText(debug_image,
                               "Accuracy: {}%".format(accuracy_percentage),
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 255, 255), 2, cv2.LINE_AA)
                    color = (0, 255, 255)
                else:
                    cv2.putText(debug_image,
                               "Accuracy: {}%".format(accuracy_percentage),
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 255), 2, cv2.LINE_AA)
                    color = (0, 0, 255)

                # Bounding box calculation
                brect = calcuate_bounding_box(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calculate_markerList(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_marker_list = pre_process_marker(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                log_csv_data(number, mode, pre_processed_marker_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_marker_list)
                if hand_sign_id == "Not applicable":  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (historyLength * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_Outerrectangle(use_brect, debug_image, brect)
                debug_image = draw_marker(debug_image, landmark_list)
                debug_image = display_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = show_historyPoint(debug_image, point_history)
        debug_image = display_info(debug_image,
                                # fps,
                                mode,
                                number)

        # Screen reflection ####
        cv2.imshow('Noises Hand_Gesture_Recognition', debug_image)

    cap.release() #release the camera
    cv2.destroyAllWindows() #close all the windows

def modeSelect(key, mode): #mode selection
    number = -1 # number
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calcuate_bounding_box(image, landmarks): # Bounding box calculation
    image_width, image_height = image.shape[1], image.shape[0] # image width, height

    landmark_array = np.empty((0, 2), int) # Keypoint

    for _, landmark in enumerate(landmarks.landmark): # Keypoint
        landmark_x = min(int(landmark.x * image_width), image_width - 1) # Keypoint x
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        marker_point = [np.array((landmark_x, landmark_y))] # Keypoint
        landmark_array = np.append(landmark_array, marker_point, axis=0) # Keypoint
    x, y, w, h = cv2.boundingRect(landmark_array) # Bounding box

    return [x, y, x + w, y + h]

def calculate_markerList(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0] # image width, height

    marker_point = [] # Keypoint
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1) # Keypoint x
        landmark_y = min(int(landmark.y * image_height), image_height - 1) # Keypoint y
        # landmark_z = landmark.z
        marker_point.append([landmark_x, landmark_y])
    return marker_point

def pre_process_marker(landmark_list):
    temp_markerList = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, marker_point in enumerate(temp_markerList):
        if index == 0: # The first point is the base point
            base_x, base_y = marker_point[0], marker_point[1] # Base point
        temp_markerList[index][0] = temp_markerList[index][0] - base_x # Relative coordinates
        temp_markerList[index][1] = temp_markerList[index][1] - base_y # Relative coordinates
    # Convert to a one-dimensional list
    temp_markerList = list(
        itertools.chain.from_iterable(temp_markerList)) # One-dimensional list

    # Normalization
    max_value = max(list(map(abs, temp_markerList))) # Maximum value

    def normalize_(n): # Normalization function
        return n / max_value # if max_value != 0 else 0

    temp_markerList = list(map(normalize_, temp_markerList))
    return temp_markerList # Normalized landmark list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0] # image width, height

    Temp_PointLog = copy.deepcopy(point_history) # Copy point history list to temporary list for processing data

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(Temp_PointLog):
        if index == 0:
            base_x, base_y = point[0], point[1] # Base point

        Temp_PointLog[index][0] = (Temp_PointLog[index][0] -
                                        base_x) / image_width
        Temp_PointLog[index][1] = (Temp_PointLog[index][1] -
                                        base_y) / image_height
    # Convert to a one-dimensional list
    Temp_PointLog = list(
        itertools.chain.from_iterable(Temp_PointLog))

    return Temp_PointLog

def log_csv_data(number, mode, landmark_list, point_history_list): # Save data to csv file
    if mode == 0: # mode 0
        pass
    if mode == 1 and (0 <= number <= 9): # mode 1 and number 0 ~ 9
        csv_path = 'model/keypoint_classifier/keypoint.csv' # csv file path
        with open(csv_path, 'a', newline="") as f:  # Open csv file
            writer = csv.writer(f)  # Write csv file
            writer.writerow([number, *landmark_list]) # Write csv file
    if mode == 2 and (0 <= number <= 9): # mode 2 and number 0 ~ 9
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f: # Open csv file
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_marker(image, marker_point): # Draw landmark
    if len(marker_point) > 0:  # If there is a landmark
        # Thumb
        cv2.line(image, tuple(marker_point[2]), tuple(marker_point[3]),
                (0, 255, 0), 6)
        cv2.line(image, tuple(marker_point[2]), tuple(marker_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[3]), tuple(marker_point[4]),
                (0, 255, 0), 6)
        cv2.line(image, tuple(marker_point[3]), tuple(marker_point[4]),
                (255, 255, 255), 2)
        # Index finger
        cv2.line(image, tuple(marker_point[5]), tuple(marker_point[6]),
                (25, 0, 0), 6)
        cv2.line(image, tuple(marker_point[5]), tuple(marker_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[6]), tuple(marker_point[7]),
                (25, 0, 0), 6)
        cv2.line(image, tuple(marker_point[6]), tuple(marker_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[7]), tuple(marker_point[8]),
                (25, 0, 0), 6)
        cv2.line(image, tuple(marker_point[7]), tuple(marker_point[8]),
                (255, 255, 255), 2)
        # Middle finger
        cv2.line(image, tuple(marker_point[9]), tuple(marker_point[10]),
                (0, 75, 0), 6)
        cv2.line(image, tuple(marker_point[9]), tuple(marker_point[10]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[10]), tuple(marker_point[11]),
                (0, 75, 0), 6)
        cv2.line(image, tuple(marker_point[10]), tuple(marker_point[11]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[11]), tuple(marker_point[12]),
                (0, 75, 0), 6)
        cv2.line(image, tuple(marker_point[11]), tuple(marker_point[12]),
                (255, 255, 255), 2)
        # Ring finger
        cv2.line(image, tuple(marker_point[13]), tuple(marker_point[14]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[13]), tuple(marker_point[14]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[14]), tuple(marker_point[15]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[14]), tuple(marker_point[15]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[15]), tuple(marker_point[16]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[15]), tuple(marker_point[16]),
                (255, 255, 255), 2)
        # Little finger
        cv2.line(image, tuple(marker_point[17]), tuple(marker_point[18]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[17]), tuple(marker_point[18]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[18]), tuple(marker_point[19]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[18]), tuple(marker_point[19]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[19]), tuple(marker_point[20]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[19]), tuple(marker_point[20]),
                (255, 255, 255), 2)
        # Palm
        cv2.line(image, tuple(marker_point[0]), tuple(marker_point[1]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[0]), tuple(marker_point[1]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[1]), tuple(marker_point[2]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[1]), tuple(marker_point[2]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[2]), tuple(marker_point[5]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[2]), tuple(marker_point[5]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[5]), tuple(marker_point[9]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[5]), tuple(marker_point[9]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[9]), tuple(marker_point[13]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[9]), tuple(marker_point[13]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[13]), tuple(marker_point[17]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[13]), tuple(marker_point[17]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(marker_point[17]), tuple(marker_point[0]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(marker_point[17]), tuple(marker_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(marker_point):
        # Draw a rectangle around the keypoint
        rect_size = 5
        cv2.rectangle(image, (landmark[0] - rect_size, landmark[1] - rect_size),
                     (landmark[0] + rect_size, landmark[1] + rect_size), (0, 255, 0), 1)
        if index == 0:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 0, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    return image

def draw_bounding_Outerrectangle(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1) # Bounding rectangle
    return image

def display_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1) # Text background

    info_text = handedness.classification[0].label[0:] # Handedness text
    if hand_sign_text != "":  # Hand sign text
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Finger gesture text removed from here
    return image

def show_historyPoint(image, point_history):
    for index, point in enumerate(point_history): # Draw point history
        if point[0] != 0 and point[1] != 0: # If point is valid
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),  # Draw circle
                      (152, 251, 152), 2) # Green
    return image

def display_info(image,
              mode,
              number):
    stringMode = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv2.putText(image, "MODE:" + stringMode[mode - 1], (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv2.LINE_AA)
    return image

if __name__ == '__main__':
    main()