import cv2
import os
import numpy as np
import glob

# === STEP 1: Prepare training data ===
def prepare_training_data(base_path):
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_name
        print(f"[INFO] Loading images for: {person_name}")

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path)

            if img is None:
                print(f"[WARNING] Skipped unreadable file: {image_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                faces.append(face_resized)
                labels.append(label_id)
                break

        label_id += 1

    return faces, np.array(labels), label_map

# === STEP 2: Predict and draw uniform green label ===
def predict_and_draw(image_path, recognizer, label_map):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to load: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces_rect) == 0:
        print(f"[WARNING] No face found in: {image_path}")
        return None

    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200, 200))
        label, confidence = recognizer.predict(face_resized)
        name = label_map[label]

        print(f"[RESULT] {os.path.basename(image_path)} → {name} (Confidence: {confidence:.2f})")

        # Add top padding for label
        top_padding = 50
        img_padded = cv2.copyMakeBorder(img, top_padding, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        y += top_padding

        # Draw face rectangle
        cv2.rectangle(img_padded, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Font and label settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        label_height = 30

        # Label background dimensions (fixed height, full width)
        label_top = y - label_height - 5
        label_bottom = y - 5
        label_left = x
        label_right = x + w
        cv2.rectangle(img_padded, (label_left, label_top), (label_right, label_bottom), (0, 255, 0), -1)

        # Center text inside the label box
        (text_width, text_height), _ = cv2.getTextSize(name, font, font_scale, thickness)
        text_x = x + (w - text_width) // 2
        text_y = label_bottom - 8
        cv2.putText(img_padded, name, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        break

    img_resized = cv2.resize(img_padded, (300, 320))
    return img_resized

# === MAIN EXECUTION ===
print("[INFO] Preparing training data...")
faces, labels, label_map = prepare_training_data(".")
print("[INFO] Training recognizer...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)
print("[INFO] Training complete!")

# === STEP 3: Predict on all images ===
images_to_display = []
image_paths = []

for person in label_map.values():
    folder = os.path.join(".", person)
    image_paths.extend(sorted(glob.glob(os.path.join(folder, "*.png"))))

for img_path in image_paths:
    img_result = predict_and_draw(img_path, recognizer, label_map)
    if img_result is not None:
        images_to_display.append(img_result)

# === STEP 4: Display in grid ===
if images_to_display:
    row_size = 3
    rows = [images_to_display[i:i + row_size] for i in range(0, len(images_to_display), row_size)]
    final_display = []

    for row_imgs in rows:
        if len(row_imgs) < row_size:
            blank = np.zeros_like(row_imgs[0])
            row_imgs += [blank] * (row_size - len(row_imgs))
        final_display.append(cv2.hconcat(row_imgs))

    grid_image = cv2.vconcat(final_display)
    cv2.imshow("All Predictions", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[INFO] No valid images to display.")