import sys
import cv2
import mediapipe as mp
import numpy as np
import pyperclip
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QPushButton, QLabel, QComboBox, QSlider)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import time

class EmojiGestureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emoji Gesture Recognition")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.status_label = QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.camera_index = 0
        self.init_camera()

        self.camera_selector = QComboBox()
        self.camera_selector.addItems([f"Camera {i}" for i in range(3)]) 
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        self.layout.insertWidget(0, QLabel("Select Camera:"))
        self.layout.insertWidget(1, self.camera_selector)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Hand Gestures", "Facial Expressions"])
        self.mode_selector.currentTextChanged.connect(self.change_mode)
        self.layout.insertWidget(2, QLabel("Select Mode:"))
        self.layout.insertWidget(3, self.mode_selector)

        # Create camera feed label
        self.camera_label = QLabel()
        self.layout.insertWidget(4, self.camera_label)

        # Initialize mode
        self.current_mode = "Hand Gestures"
        
        # Setup emoji mappings
        self.hand_gesture_emojis = {
            "ok": "👌",
            "thumbs_up": "👍",
            "thumbs_down": "👎",
            "peace": "✌️",
            "fist": "✊",
            "point_up": "👆",
            "point_down": "👇",
            "point_left": "👈",
            "point_right": "👉",
            "wave": "👋",
            "rock": "🤘",
            "love": "🤟",
            "call": "🤙",
            "clap": "👏",
            "open_hand": "✋",
            "pinch": "🤏",
        }
        
        self.facial_expression_emojis = {
            "neutral": "😐",
            "slight_smile": "🙂",
            "smile": "😊",
            "big_smile": "😃",
            "laugh": "😄",
            "laugh_tears": "😂",
            "surprise": "😮",
            "shock": "😱",
            "sad": "😢",
            "cry": "😭",
            "wink": "😉",
            "tongue": "😛",
            "kiss": "😘",
        }

        # Last detection time to prevent rapid-fire detections
        self.last_detection_time = 0
        self.detection_cooldown = 1.0  # seconds

        # Start timer for video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms refresh rate

    def change_mode(self, mode):
        self.current_mode = mode
        self.status_label.setText(f"Mode changed to: {mode}")

    def update_frame(self):
        """Update the camera frame with error handling"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.status_label.setText("Error: Camera not available")
            return

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.status_label.setText("Error: Failed to read camera frame")
                return

            # Convert BGR to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.current_mode == "Hand Gestures":
                self.process_hand_gestures(frame)
            else:
                self.process_facial_expressions(frame)

            # Convert frame to Qt format and display
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                640, 480, Qt.KeepAspectRatio))
        except Exception as e:
            self.status_label.setText(f"Error processing frame: {str(e)}")

    def process_hand_gestures(self, frame):
        results = self.hands.process(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get current time
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_cooldown:
                    return

                # Detect various gestures
                gesture = self.detect_hand_gesture(hand_landmarks)
                if gesture and gesture in self.hand_gesture_emojis:
                    self.copy_emoji(self.hand_gesture_emojis[gesture])
                    self.status_label.setText(f"Detected: {gesture}")
                    self.last_detection_time = current_time

    def detect_hand_gesture(self, landmarks):
        # Get all finger positions
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_base = landmarks.landmark[2]
        index_tip = landmarks.landmark[8]
        index_mid = landmarks.landmark[7]
        index_base = landmarks.landmark[6]
        index_mcp = landmarks.landmark[5]
        middle_tip = landmarks.landmark[12]
        middle_mid = landmarks.landmark[11]
        middle_base = landmarks.landmark[10]
        middle_mcp = landmarks.landmark[9]
        ring_tip = landmarks.landmark[16]
        ring_mid = landmarks.landmark[15]
        ring_base = landmarks.landmark[14]
        ring_mcp = landmarks.landmark[13]
        pinky_tip = landmarks.landmark[20]
        pinky_mid = landmarks.landmark[19]
        pinky_base = landmarks.landmark[18]
        pinky_mcp = landmarks.landmark[17]
        wrist = landmarks.landmark[0]

        # Helper functions
        def finger_extended(tip, mid, base, mcp):
            vertical_extension = tip.y < mid.y < base.y
            finger_straightness = abs((tip.x - base.x) / (tip.y - base.y + 1e-6)) < 0.5
            return vertical_extension and finger_straightness

        def finger_bent(tip, mid, base):
            return tip.y > mid.y

        def calc_angle(p1, p2, p3):
            v1 = [p1.x - p2.x, p1.y - p2.y]
            v2 = [p3.x - p2.x, p3.y - p2.y]
            angle = np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])
            return np.degrees(angle)

        def distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        def is_horizontal(p1, p2, threshold=0.15):  
            return abs(p1.y - p2.y) < threshold

        def is_vertical(p1, p2, threshold=0.15):  
            return abs(p1.x - p2.x) < threshold

        def is_pointing(tip, mid, base, direction='right'):
            # Check if finger is extended and pointing in the specified direction
            if direction in ['right', 'left']:
                # For horizontal pointing, check y-alignment and x-direction
                is_aligned = abs(tip.y - base.y) < 0.15
                if direction == 'right':
                    return is_aligned and (tip.x > base.x) and (mid.x > base.x)
                else:
                    return is_aligned and (tip.x < base.x) and (mid.x < base.x)
            else:  # up or down
                # For vertical pointing, check x-alignment and y-direction
                is_aligned = abs(tip.x - base.x) < 0.15
                if direction == 'up':
                    return is_aligned and (tip.y < base.y) and (mid.y < base.y)
                else:
                    return is_aligned and (tip.y > base.y) and (mid.y > base.y)

        # Check for OK sign
        if (distance(thumb_tip, index_tip) < 0.05 and
            finger_extended(middle_tip, middle_mid, middle_base, middle_mcp) and
            finger_extended(ring_tip, ring_mid, ring_base, ring_mcp) and
            finger_extended(pinky_tip, pinky_mid, pinky_base, pinky_mcp)):
            return "ok"

        # Check for thumbs up
        if (thumb_tip.y < thumb_base.y and
            finger_bent(index_tip, index_mid, index_base) and
            finger_bent(middle_tip, middle_mid, middle_base) and
            finger_bent(ring_tip, ring_mid, ring_base) and
            finger_bent(pinky_tip, pinky_mid, pinky_base) and
            is_vertical(thumb_tip, thumb_base)):
            return "thumbs_up"

        # Check for thumbs down
        if (thumb_tip.y > thumb_base.y and
            finger_bent(index_tip, index_mid, index_base) and
            finger_bent(middle_tip, middle_mid, middle_base) and
            finger_bent(ring_tip, ring_mid, ring_base) and
            finger_bent(pinky_tip, pinky_mid, pinky_base) and
            is_vertical(thumb_tip, thumb_base)):
            return "thumbs_down"

        # Check for pointing right
        if (is_pointing(index_tip, index_mid, index_base, 'right') and
            finger_bent(middle_tip, middle_mid, middle_base) and
            finger_bent(ring_tip, ring_mid, ring_base) and
            finger_bent(pinky_tip, pinky_mid, pinky_base)):
            return "point_right"

        # Check for pointing left
        if (is_pointing(index_tip, index_mid, index_base, 'left') and
            finger_bent(middle_tip, middle_mid, middle_base) and
            finger_bent(ring_tip, ring_mid, ring_base) and
            finger_bent(pinky_tip, pinky_mid, pinky_base)):
            return "point_left"

        # Check for pointing up
        if (is_pointing(index_tip, index_mid, index_base, 'up') and
            finger_bent(middle_tip, middle_mid, middle_base) and
            finger_bent(ring_tip, ring_mid, ring_base) and
            finger_bent(pinky_tip, pinky_mid, pinky_base)):
            return "point_up"

        # Check for pointing down
        if (is_pointing(index_tip, index_mid, index_base, 'down') and
            finger_bent(middle_tip, middle_mid, middle_base) and
            finger_bent(ring_tip, ring_mid, ring_base) and
            finger_bent(pinky_tip, pinky_mid, pinky_base)):
            return "point_down"

        # Detect peace sign (V shape with index and middle fingers)
        if (finger_extended(index_tip, index_mid, index_base, index_mcp) and
            finger_extended(middle_tip, middle_mid, middle_base, middle_mcp) and
            not finger_extended(ring_tip, ring_mid, ring_base, ring_mcp) and
            not finger_extended(pinky_tip, pinky_mid, pinky_base, pinky_mcp) and
            distance(index_tip, middle_tip) > 0.1):  # Ensure fingers are spread
            return "peace"

        # Detect fist
        if (all(not finger_extended(tip, mid, base, mcp) for tip, mid, base, mcp in [
            (index_tip, index_mid, index_base, index_mcp),
            (middle_tip, middle_mid, middle_base, middle_mcp),
            (ring_tip, ring_mid, ring_base, ring_mcp),
            (pinky_tip, pinky_mid, pinky_base, pinky_mcp)
        ]) and thumb_tip.x < index_mcp.x):  # Thumb tucked in
            return "fist"

        # Detect pointing directions
        if (finger_extended(index_tip, index_mid, index_base, index_mcp) and
            all(not finger_extended(tip, mid, base, mcp) for tip, mid, base, mcp in [
                (middle_tip, middle_mid, middle_base, middle_mcp),
                (ring_tip, ring_mid, ring_base, ring_mcp),
                (pinky_tip, pinky_mid, pinky_base, pinky_mcp)
            ])):
            # Calculate pointing angle
            point_angle = calc_angle(index_tip, index_base, wrist)
            if -30 <= point_angle <= 30:
                return "point_right"
            elif 150 <= point_angle or point_angle <= -150:
                return "point_left"
            elif 60 <= point_angle <= 120:
                return "point_down"
            elif -120 <= point_angle <= -60:
                return "point_up"

        # Detect wave (fingers extended and hand tilted)
        if (all(finger_extended(tip, mid, base, mcp) for tip, mid, base, mcp in [
            (index_tip, index_mid, index_base, index_mcp),
            (middle_tip, middle_mid, middle_base, middle_mcp),
            (ring_tip, ring_mid, ring_base, ring_mcp),
            (pinky_tip, pinky_mid, pinky_base, pinky_mcp)
        ]) and abs(calc_angle(index_tip, wrist, pinky_tip)) > 30):
            return "wave"

        # Detect rock sign
        if (finger_extended(index_tip, index_mid, index_base, index_mcp) and
            not finger_extended(middle_tip, middle_mid, middle_base, middle_mcp) and
            not finger_extended(ring_tip, ring_mid, ring_base, ring_mcp) and
            finger_extended(pinky_tip, pinky_mid, pinky_base, pinky_mcp) and
            thumb_tip.x > index_base.x):  # Thumb out
            return "rock"

        # Detect love sign
        if (finger_extended(index_tip, index_mid, index_base, index_mcp) and
            not finger_extended(middle_tip, middle_mid, middle_base, middle_mcp) and
            not finger_extended(ring_tip, ring_mid, ring_base, ring_mcp) and
            finger_extended(pinky_tip, pinky_mid, pinky_base, pinky_mcp) and
            thumb_tip.x < index_base.x):  # Thumb in
            return "love"

        # Detect call sign
        if (not finger_extended(index_tip, index_mid, index_base, index_mcp) and
            not finger_extended(middle_tip, middle_mid, middle_base, middle_mcp) and
            not finger_extended(ring_tip, ring_mid, ring_base, ring_mcp) and
            finger_extended(pinky_tip, pinky_mid, pinky_base, pinky_mcp) and
            thumb_tip.x > pinky_base.x):  # Thumb out
            return "call"

        # Detect clap (hands close together with fingers aligned)
        if (all(finger_extended(tip, mid, base, mcp) for tip, mid, base, mcp in [
            (index_tip, index_mid, index_base, index_mcp),
            (middle_tip, middle_mid, middle_base, middle_mcp),
            (ring_tip, ring_mid, ring_base, ring_mcp),
            (pinky_tip, pinky_mid, pinky_base, pinky_mcp)
        ]) and abs(index_tip.x - pinky_tip.x) < 0.1):  # Fingers close together
            return "clap"

        # Detect open hand
        if (all(finger_extended(tip, mid, base, mcp) for tip, mid, base, mcp in [
            (index_tip, index_mid, index_base, index_mcp),
            (middle_tip, middle_mid, middle_base, middle_mcp),
            (ring_tip, ring_mid, ring_base, ring_mcp),
            (pinky_tip, pinky_mid, pinky_base, pinky_mcp)
        ]) and thumb_tip.x > index_base.x):  # Thumb out
            return "open_hand"

        # Detect pinch
        if (distance(thumb_tip, index_tip) < 0.05 and  # Thumb and index close
            all(finger_extended(tip, mid, base, mcp) for tip, mid, base, mcp in [
                (middle_tip, middle_mid, middle_base, middle_mcp),
                (ring_tip, ring_mid, ring_base, ring_mcp),
                (pinky_tip, pinky_mid, pinky_base, pinky_mcp)
            ])):
            return "pinch"

        return None

    def process_facial_expressions(self, frame):
        results = self.face_mesh.process(frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None)
                
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_cooldown:
                    return

                expression = self.detect_facial_expression(face_landmarks)
                if expression and expression in self.facial_expression_emojis:
                    self.copy_emoji(self.facial_expression_emojis[expression])
                    self.status_label.setText(f"Detected: {expression}")
                    self.last_detection_time = current_time

    def detect_facial_expression(self, landmarks):
        def get_point(idx):
            return landmarks.landmark[idx]

        # Eyes
        left_eye_top = get_point(386)
        left_eye_bottom = get_point(374)
        right_eye_top = get_point(159)
        right_eye_bottom = get_point(145)
        
        # Mouth points for teeth visibility and general shape
        mouth_top = get_point(13)        # Upper lip
        mouth_bottom = get_point(14)     # Lower lip
        mouth_top_inner = get_point(12)  # Inner upper lip
        mouth_bottom_inner = get_point(15)# Inner lower lip
        
        # Mouth corners for smile/sad detection
        mouth_left = get_point(61)
        mouth_right = get_point(291)
        
        # Additional points for lip pursing (kiss)
        upper_lip_outer = get_point(0)
        lower_lip_outer = get_point(17)
        
        # Face contraction points (for sad)
        cheek_left = get_point(234)      # Left cheek
        cheek_right = get_point(454)     # Right cheek
        forehead = get_point(10)         # Forehead
        chin = get_point(152)            # Chin
        
        def calc_distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        # Calculate eye openness
        left_eye_height = calc_distance(left_eye_top, left_eye_bottom)
        right_eye_height = calc_distance(right_eye_top, right_eye_bottom)
        eye_openness = (left_eye_height + right_eye_height) / 2
        
        # Calculate mouth metrics
        mouth_height = calc_distance(mouth_top, mouth_bottom)  # Total mouth height
        mouth_inner_height = calc_distance(mouth_top_inner, mouth_bottom_inner)  # Inner mouth (teeth visibility)
        lip_pucker = calc_distance(upper_lip_outer, lower_lip_outer)  # For kiss detection
        
        # Calculate face contraction (for sad)
        face_height = calc_distance(forehead, chin)
        face_width = calc_distance(cheek_left, cheek_right)
        face_contraction = face_width / face_height  # Lower value means more contracted
        
        # Calculate lip corner positions for smile/sad
        mouth_corner_y = (mouth_left.y + mouth_right.y) / 2
        mouth_center_y = (mouth_top.y + mouth_bottom.y) / 2
        lip_curve = mouth_corner_y - mouth_center_y

        # Debug values
        print(f"Eye openness: {eye_openness:.3f}")
        print(f"Mouth height: {mouth_height:.3f}")
        print(f"Face contraction: {face_contraction:.3f}")
        print(f"Lip curve: {lip_curve:.3f}")

        # Expression detection based on user's definitions
        
        # 1. Check for shock/surprise (eyes AND mouth open)
        if eye_openness > 0.045 and mouth_height > 0.08:  # Adjusted thresholds
            return "shock"
            
        # 2. Check for sad (whole face contracted)
        if face_contraction < 1.2 and lip_curve > 0.005:  # Face contracted and slight downturn of mouth
            return "sad"
            
        # 3. Check for kiss (pursed lips)
        if lip_pucker < 0.02:
            return "kiss"
            
        # 4. Check for smile (teeth visible)
        if mouth_inner_height > 0.03:
            return "smile"
            
        # 5. Default to neutral if no other expression detected
        return "neutral"

    def copy_emoji(self, emoji):
        pyperclip.copy(emoji)
        
    def init_camera(self):
        try:
            if hasattr(self, 'cap'):
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)  
            
            if not self.cap.isOpened():
                self.status_label.setText(f"Error: Could not open camera {self.camera_index}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.status_label.setText(f"Camera {self.camera_index} initialized successfully")
            return True
        except Exception as e:
            self.status_label.setText(f"Error initializing camera: {str(e)}")
            return False

    def change_camera(self, index):
        self.camera_index = index
        if self.init_camera():
            self.status_label.setText(f"Switched to Camera {index}")
        else:
            self.status_label.setText(f"Failed to switch to Camera {index}")

    def closeEvent(self, event):
        self.cap.release()
        self.hands.close()
        self.face_mesh.close()
        event.accept()

def main():
    try:
        print("Starting application...")
        app = QApplication(sys.argv)
        print("Created QApplication")
        
        window = EmojiGestureApp()
        print("Created main window")
        
        window.show()
        print("Window shown")
        
        return app.exec_()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return 1

if __name__ == '__main__':
    print("Python version:", sys.version)
    print("OpenCV version:", cv2.__version__)
    print("MediaPipe version:", mp.__version__)
    print("Starting main function...")
    sys.exit(main())
