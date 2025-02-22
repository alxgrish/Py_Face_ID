import pyodbc
import numpy as np
import os
import cv2
import face_recognition
from tkinter import messagebox, simpledialog

# Параметры подключения к 1С через ODBC
dsn = '1C_Connection'  # Название источника данных (DSN), настроенного в ODBC
user = 'your_user'
password = 'your_password'

# Подключение к базе данных 1С через ODBC
def connect_to_1c():
    connection = pyodbc.connect(f'DSN={dsn};UID={user};PWD={password}')
    return connection

# Функция загрузки данных студентов из 1С
def load_students_from_1c():
    connection = connect_to_1c()
    cursor = connection.cursor()
    
    # SQL-запрос для получения данных о студентах
    cursor.execute("SELECT Student_FIO, Curator_FIO, Student_Login, Embedding_Code FROM Students")
    
    students = []
    for row in cursor.fetchall():
        student_data = {
            'student_fio': row.Student_FIO,
            'curator_fio': row.Curator_FIO,
            'student_login': row.Student_Login,
            'embedding_code': row.Embedding_Code
        }
        students.append(student_data)
    
    cursor.close()
    connection.close()
    
    return students

# Функция сохранения эмбеддинга студента
def save_student_embedding(student_fio, embedding_code):
    npy_path = os.path.join("embeddings", f"{student_fio}.npy")
    
    # Проверка, существует ли файл эмбеддинга
    if os.path.exists(npy_path):
        existing_encodings = np.load(npy_path)
        updated_encodings = np.vstack([existing_encodings, embedding_code])
    else:
        updated_encodings = np.array([embedding_code])
    
    np.save(npy_path, updated_encodings)

# Функция распознавания лиц с загрузкой эмбеддингов из 1С
def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    known_encodings = []
    known_names = []

    # Загрузка студентов и эмбеддингов
    students = load_students_from_1c()
    for student in students:
        known_names.append(student['student_fio'])
        # Код эмбеддинга — это просто пример, для реальной ситуации нужно получить реальные эмбеддинги
        # Например, используйте заранее сгенерированные эмбеддинги для каждого студента
        known_encodings.append(np.array([float(x) for x in student['embedding_code'].split(',')]))
    
    known_encodings = np.array(known_encodings)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Неизвестный"
            if known_encodings.size > 0:  # Проверяем, что массив не пустой
                face_encoding = np.array(face_encoding).reshape(1, -1)  # Преобразуем в 2D
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(distances)

                if distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]

                    # Обновляем эмбеддинг для найденного лица
                    save_student_embedding(name, face_encoding[0])
# Рисуем прямоугольник вокруг лица и имя
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Отображаем изображение
        cv2.imshow("Распознавание лиц", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Пример использования
recognize_faces()
