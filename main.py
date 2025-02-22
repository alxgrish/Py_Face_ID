
import cv2
import face_recognition
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

DATA_DIR = "embeddings"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Функция для сохранения эмбеддинга
def save_embedding(name, encoding):
    filepath = os.path.join(DATA_DIR, f"{name}.npy")
    np.save(filepath, encoding)
    messagebox.showinfo("Успешно", f"Эмбеддинг сохранён: {filepath}")
    update_user_list()

# Функция загрузки всех сохранённых лиц
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

        cv2.putText(frame, "press 'S' for save", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face scan", frame)

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

    # Ввод имени
    name = simpledialog.askstring("Сканирование", "Введите имя:")
    if not name:
        messagebox.showerror("Ошибка", "Имя не может быть пустым")
        return

    # Сохранение эмбеддинга
    save_embedding(name, face_encodings[0])

# Функция распознавания лиц
def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    known_encodings, known_names = load_known_faces()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.6:  # Порог для уверенности
                    name = known_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
    filepath = os.path.join(DATA_DIR, f"{name}.npy")

    if os.path.exists(filepath):
        os.remove(filepath)
        messagebox.showinfo("Удалено", f"Пользователь {name} удалён")
        update_user_list()
    else:
        messagebox.showerror("Ошибка", "Файл не найден")

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