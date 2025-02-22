import cv2
import face_recognition
import face_recognition_models


# Загружаем или создаем список эмбеддингов лиц (для начала пустой)
known_face_encodings = []
known_face_names = []

# Открываем камеру
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Не удалось захватить изображение")
        break

    # Конвертируем изображение в формат RGB (Face Recognition работает в RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Находим все лица и их эмбеддинги на текущем кадре
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Сравниваем текущее лицо с уже сохраненными эмбеддингами
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "New Face"

        # Если нашли совпадение, то это уже сохраненное лицо
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            cv2.putText(frame, "Successful", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Если лицо не найдено, добавляем в базу эмбеддинг этого лица
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Пишем имя или "Unknown" на изображении
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Отображаем результат
    cv2.imshow("Face Recognition", frame)

    # Выход из программы при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

