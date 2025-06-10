
import locale
import cv2
import face_recognition
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import time
import serial

locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')


serialPort = serial.Serial(
    port="COM4", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)


DATA_DIR = "embeddings"
camera_active = False

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
# Функция для сохранения или обновления эмбеддингов
def save_embedding(name, new_encoding):
    npy_path = os.path.join(DATA_DIR, f"{name}.npy")
    txt_path = os.path.join(DATA_DIR, f"{name} text.txt")

    if os.path.exists(npy_path):
        existing_encodings = np.load(npy_path)
        updated_encodings = np.vstack([existing_encodings, new_encoding])
    else:
        updated_encodings = np.array([new_encoding])

    np.save(npy_path, updated_encodings)

    # Дополнительный вывод в .txt
    with open(txt_path, "w") as f:
        for encoding in updated_encodings:
            f.write(",".join(map(str, encoding)) + "\n")

    update_user_list()

# Функция загрузки всех сохранённых лиц
def load_known_faces():
    known_encodings = []
    known_names = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            path = os.path.join(DATA_DIR, file)
            encodings = np.load(path)

            # Усредняем несколько эмбеддингов
            mean_encoding = np.mean(encodings, axis=0)
            known_encodings.append(mean_encoding)
            known_names.append(os.path.splitext(file)[0])

    return known_encodings, known_names


# Функция обновления списка пользователей
def update_user_list():
    users_list.delete(0, tk.END)
    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            users_list.insert(tk.END, os.path.splitext(file)[0])

# Функция сканирования лица
def scan_face():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        cv2.putText(frame, "Tap 'S' For save", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Scanning face", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Определение эмбеддинга
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)

    if not face_encodings:
        messagebox.showerror("Ошибка", "Лицо не найдено!")
        return
    if face_encodings:
        serialPort.write(1)
    else:
        serialPort.write(0)

    # Ввод имени
    name = simpledialog.askstring("Сканирование", "Введите имя:")
    if not name:
        messagebox.showerror("Ошибка", "Имя не может быть пустым")
        return

    # Сохранение или обновление эмбеддинга
    save_embedding(name, face_encodings[0])

# Функция распознавания лиц с высокой частотой кадров и обновлением эмбеддингов раз в 30 секунд
def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    known_encodings, known_names = load_known_faces()

    if not known_encodings:  # Проверяем, есть ли загруженные лица
        messagebox.showerror("Ошибка", "Нет загруженных эмбеддингов!")
        video_capture.release()
        return

    known_encodings = np.array(known_encodings)  # Убедимся, что массив двухмерный
    if known_encodings.ndim == 1:
        known_encodings = known_encodings.reshape(1, -1)

    prev_frame_time = 0  # Время предыдущего кадра
    new_frame_time = 0  # Время текущего кадра
    last_update_time = time.time()  # Время последнего обновления эмбеддингов

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
# Уменьшаем разрешение для повышения производительности
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unikown"
            if known_encodings.size > 0:  # Проверяем, что массив не пустой
                face_encoding = np.array(face_encoding).reshape(1, -1)  # Преобразуем в 2D
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.4:
                    name = known_names[best_match_index]

                    # Проверяем, прошло ли 30 секунд с последнего обновления
                    current_time = time.time()
                    if current_time - last_update_time >= 10:
                        save_embedding(name, face_encoding[0])
                        last_update_time = current_time  # Обновляем временную метку

            # Рисуем прямоугольник вокруг лица и имя
            cv2.rectangle(frame, (left * 2, top * 2), (right * 2, bottom * 2), (0, 255, 0), 2)
            cv2.putText(frame, name, (left * 2, top * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Отображаем ключевые точки
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
            for face_landmarks in face_landmarks_list:
                for facial_feature in face_landmarks.values():
                    for point in facial_feature:
                        cv2.circle(frame, point, 1, (0, 255, 0), -1)

        # Подсчёт FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Показываем FPS на экране
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Отображаем изображение
        cv2.imshow("Распознавание лиц", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Функция удаления пользователя
def delete_user():
    selected = users_list.curselection()
    if not selected:
        messagebox.showerror("Ошибка", "Выберите пользователя для удаления")
        return

    name = users_list.get(selected[0])
    npy_path = os.path.join(DATA_DIR, f"{name}.npy")
    txt_path = os.path.join(DATA_DIR, f"{name}.txt")

    if os.path.exists(npy_path):
        os.remove(npy_path)
    if os.path.exists(txt_path):
        os.remove(txt_path)

    messagebox.showinfo("Удалено", f"Пользователь {name} удалён")
    update_user_list()

# Графический интерфейс
root = tk.Tk()
root.title("Распознавание лиц")

frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="Выберите действие", font=("Arial", 14))
label.pack()

scan_button = tk.Button(frame, text="Сканировать лицо", command=scan_face, font=("Arial", 12))
scan_button.pack(pady=5)

recognize_button = tk.Button(frame, text="Распознать лица", command=recognize_faces, font=("Arial", 12))
recognize_button.pack(pady=5)

delete_button = tk.Button(frame, text="Удалить выбранного пользователя", command=delete_user, font=("Arial", 12))
delete_button.pack(pady=5)

# Список сохранённых пользователей
users_list = tk.Listbox(root, height=10, width=30)
users_list.pack(pady=10)
update_user_list()

root.mainloop()

