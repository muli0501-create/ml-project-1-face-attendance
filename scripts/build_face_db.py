import face_recognition
import os, pickle

face_db = {}   # { '姓名': [encoding1, encoding2, ...] }

for person_name in os.listdir('face_db'):
    person_dir = os.path.join('face_db', person_name)
    if not os.path.isdir(person_dir): continue

    encodings = []
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        image = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(image)
        if encs:
            encodings.append(encs[0])

    if encodings:
        face_db[person_name] = encodings
        print(f'✓ {person_name}: {len(encodings)} 张照片已入库')

with open('face_db/encodings.pkl', 'wb') as f:
    pickle.dump(face_db, f)
print(f'人脸库构建完成，共 {len(face_db)} 人')

