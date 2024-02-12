from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import pickle
import joblib
import pyttsx3
from pygame import mixer
from datetime import datetime, time
from scipy.special import expit
from threading import Thread
import time as t

class FaceFeaturesApp:
    def __init__(self):
        self.mp_face_mes = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mes.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.emotion_model = pickle.load(open('/home/ervinpicardal/Downloads/features/emotion_model.pkl', 'rb'))
        self.pca_model = joblib.load('/home/ervinpicardal/Downloads/features/pca_model.pkl')
        self.mesh_points = None
        self.emotion_labels = ['angry', 'sad', 'happy']
        self.emotion_scores = [0]*3
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.stage = None
        self.stopped = False 
        self.set_alarm_time(10, 30) 
        self.alarm_triggered = False
        self.stopped = False
        self.angle = 121
        mixer.init()

        self.prev_time = 0

        self.engine.setProperty('voice', self.voices[1].id)
        self.engine.setProperty('rate', 125)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        

    def set_alarm_time(self, hour, minute):
        self.alarm_time = time(hour, minute)

    def check_alarm(self):
        current_time = datetime.now().time()
        if current_time.hour == self.alarm_time.hour and current_time.minute == self.alarm_time.minute:
            if not self.alarm_triggered:
                self.trigger_alarm()
                self.alarm_triggered = True  
        else:
            self.alarm_triggered = False

    def trigger_alarm(self):
        mixer.music.load("/home/ervinpicardal/Downloads/features/alarm.wav")
        mixer.music.play()

    def calculate_angle(self, a, b, c):
        a = np.array(a)  
        b = np.array(b)  
        c = np.array(c)  
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
            np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360-angle
        return angle

    def transform_to_zero_one_numpy(self, arr):
        if len(arr) == 0:
            return arr
        min_val = np.min(arr)
        max_val = np.max(arr)
        if min_val == max_val:
            return np.zeros_like(arr)
        value_range = max_val - min_val
        transformed_arr = (arr - min_val) / value_range
        return transformed_arr

    def predict_emotion(self):
        global mesh_points
        if self.mesh_points is None:
            return None
        nose_tip = self.mesh_points[4]
        forehead = self.mesh_points[151]
        mesh_norm = self.mesh_points - nose_tip
        scale_factor = np.linalg.norm(forehead - nose_tip)
        if np.isclose(scale_factor, 0):
            scale_factor = 1e-6
        mesh_norm = np.divide(mesh_norm, scale_factor)
        landmarks_flat = mesh_norm.flatten()
        landmarks_transformed = self.pca_model.transform([landmarks_flat])
        pred = self.emotion_model.predict_proba(landmarks_transformed)[0]
        emotion_scores_noisy = self.transform_to_zero_one_numpy(pred)
        for score in range(len(self.emotion_scores)):
            emotion_scores_noisy[score] = expit(10 * (emotion_scores_noisy[score] - 0.5))
            self.emotion_scores[score] = self.emotion_scores[score] * 0.9 + emotion_scores_noisy[score] * 0.1
        pred_index = np.argmax(self.emotion_scores)
        return self.emotion_labels[pred_index]

    def sad(self):
        emotion_scores_rounded = [round(score, 2) for score in self.emotion_scores]
        #print(emotion_scores_rounded)
        if 0.80 <= emotion_scores_rounded[1] <= 0.86:
            mixer.music.load("/home/ervinpicardal/Downloads/features/MarriedLife.mp3")
            mixer.music.play()
            pass

    def fall_check(self, angle):
        # if angle >= 140:
        #     self.stage = "stand"
        if angle < 120:
            # self.stage = "fall"
            curr_time = t.time()
            if(curr_time - self.prev_time > 0.3):
                print("Fall detected")
                mixer.music.load("/home/ervinpicardal/Downloads/features/fall.mp3")
                mixer.music.play()
            self.prev_time = curr_time
    
    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            frame_facemesh = frame.copy()
            empty = np.zeros(frame.shape, dtype=np.uint8)
            H, W, _ = frame_facemesh.shape
            rgb_image = cv2.cvtColor(frame_facemesh, cv2.COLOR_BGR2RGB)
            results_mesh = self.face_mesh.process(rgb_image)
            if results_mesh.multi_face_landmarks:
                self.mesh_points = np.array([np.multiply([p.x, p.y], [W, H]).astype(int) for p in results_mesh.multi_face_landmarks[0].landmark])
                emotion = self.predict_emotion()
                if emotion:
                    cv2.putText(frame_facemesh, emotion, (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(empty, emotion, (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.sad()

            with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
                results = pose.process(frame)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    angle = self.calculate_angle(shoulder, hip, knee)
                    
                    self.fall_check(angle)
                    
            self.check_alarm()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceFeaturesApp()
    app.run()