import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer
import time

# Initialize Pygame mixer for audio alerts
mixer.init()
mixer.music.load("music.wav")

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # Vertical
    B = distance.euclidean(mouth[4], mouth[8])   # Vertical
    C = distance.euclidean(mouth[0], mouth[6])   # Horizontal
    mar = (A + B) / (2.0 * C)
    return mar

# Parameters
EYE_THRESH = 0.25  # EAR threshold for drowsiness
MOUTH_THRESH = 0.6  # MAR threshold for yawning
FRAME_CHECK = 40    # Consecutive frames required for drowsiness alert
ALERT_DISPLAY_TIME = 3  # Seconds to display alerts
ALARM_COOLDOWN = 5  # Seconds between alarms

# Initialize counters and states
drowsy_flag = 0
yawn_state = 0  # 0: no yawn, 1: yawn open, 2: yawn closed
yawn_count = 0
last_alarm_time = time.time()
last_alert_time = 0
active_alert = None

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']

# Load dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Eye detection
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Mouth detection
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # Draw contours for visualization
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

        # Drowsiness detection
        if ear < EYE_THRESH:
            drowsy_flag += 1
            if drowsy_flag >= FRAME_CHECK and (time.time() - last_alarm_time) > ALARM_COOLDOWN:
                active_alert = "DROWSY ALERT! Eyes Closed!"
                last_alert_time = time.time()
                mixer.music.play()
                last_alarm_time = time.time()
        else:
            drowsy_flag = 0  # Reset flag if EAR is above threshold

        # Yawning detection
        if mar > MOUTH_THRESH:
            if yawn_state == 0:  # Mouth opens
                yawn_state = 1
        elif yawn_state == 1:  # Mouth closes after being open
            yawn_state = 0
            yawn_count += 1
            if yawn_count == 1:
                active_alert = "YAWNING ALERT! First Yawn Detected!"
                last_alert_time = time.time()
            elif yawn_count >= 2 and (time.time() - last_alarm_time) > ALARM_COOLDOWN:
                active_alert = "YAWNING ALERT! Multiple Yawns! Beeping!"
                last_alert_time = time.time()
                mixer.music.play()
                yawn_count = 0  # Reset after multiple yawns

    # Display active alerts
    if active_alert and (time.time() - last_alert_time) < ALERT_DISPLAY_TIME:
        cv2.putText(frame, active_alert, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        active_alert = None

    # Display results
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
