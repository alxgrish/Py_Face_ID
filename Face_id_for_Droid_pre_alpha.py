from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
import cv2
import face_recognition
import numpy as np
import os
from threading import Thread

# Папка для эмбеддингов
DATA_DIR = "faces"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Функция загрузки эмбеддингов
def load_known_faces():
    known_encodings = []
    known_names = []
    
    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):  
            path = os.path.join(DATA_DIR, file)
            encoding = np.load(path)
            known_encodings.append(encoding)
            known_names.append(os.path.splitext(file)[0])

    return known_encodings, known_names

# Функция сохранения лица
def save_face(name, face_encoding):
    path = os.path.join(DATA_DIR, f"{name}.npy")

    if os.path.exists(path):
        existing_encodings = np.load(path)
        face_encoding = np.vstack([existing_encodings, face_encoding])
    
    np.save(path, face_encoding)

# Класс интерфейса Kivy
class FaceRecognitionApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.label = Label(text="Система распознавания лиц", font_size=20)
        layout.add_widget(self.label)

        self.name_input = TextInput(hint_text="Введите имя", size_hint=(1, 0.1))
        layout.add_widget(self.name_input)

        self.image_widget = Image()
        layout.add_widget(self.image_widget)

        self.scan_button = Button(text="Сканировать лицо", size_hint=(1, 0.2))
        self.scan_button.bind(on_press=self.start_scan)
        layout.add_widget(self.scan_button)

        self.recognize_button = Button(text="Распознать лица", size_hint=(1, 0.2))
        self.recognize_button.bind(on_press=self.start_recognition)
        layout.add_widget(self.recognize_button)

        self.manage_button = Button(text="Управление пользователями", size_hint=(1, 0.2))
        self.manage_button.bind(on_press=self.manage_users)
        layout.add_widget(self.manage_button)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 0)  # Переворот потока видео
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = self.image_widget.texture
            if texture is None:
                from kivy.graphics.texture import Texture
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]))
                self.image_widget.texture = texture
            self.image_widget.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    def start_scan(self, instance):
        Thread(target=self.scan_face).start()

    def start_recognition(self, instance):
        Thread(target=self.recognize_faces).start()

    def scan_face(self):
        name = self.name_input.text.strip()
        if not name:
            self.label.text = "Введите имя перед сканированием!"
            return

        self.label.text = "Идет сканирование..."
        ret, frame = self.capture.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            save_face(name, face_encodings[0])
            self.label.text = f"Лицо {name} сохранено!"

    def recognize_faces(self):
        known_encodings, known_names = load_known_faces()
        if not known_encodings:
            self.label.text = "Нет загруженных лиц!"
            return

        ret, frame = self.capture.read()
        if not ret:
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Неизвестный"
            distances = face_recognition.face_distance(np.array(known_encodings), face_encoding)
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
                self.label.text = f"User {name} Detected"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            face_landmarks = face_recognition.face_landmarks(rgb_frame, [face_encoding])
            if face_landmarks:
                for facial_feature in face_landmarks[0]:
                    for point in face_landmarks[0][facial_feature]:
                        cv2.circle(frame, point, 2, (0, 0, 255), -1)

        cv2.imwrite("frame.png", frame)
        self.image_widget.source = "frame.png"
        self.image_widget.reload()

    def manage_users(self, instance):
        self.label.text = "Функция управления пользователями в разработке."

if __name__ == "__main__":
    FaceRecognitionApp().run()
