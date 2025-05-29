import cv2
import numpy as np
from keras.models import load_model

# Load trained CNN model
model = load_model('CNNmodel.h5')

# List of labels (29 classes)
classes = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']  # A-Z + del, nothing, space

def keras_predict(model, image):
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0  # Normalize
    prediction = model.predict(image, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    return confidence, predicted_class

def keras_process_image(img):
    img = cv2.resize(img, (64, 64))  # Resize to match training input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    img = np.expand_dims(img, axis=0)  # Shape: (1, 64, 64, 3)
    return img

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def main():
    cam_capture = cv2.VideoCapture(0)
    cam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    x, y, w, h = 400, 150, 400, 400  # ROI position and size

    sentence = ""
    last_letter = ""
    count_same_letter = 0
    CONFIDENCE_THRESHOLD = 0.8
    STABILITY_THRESHOLD = 5  # Stable frames before accepting prediction

    print("📸 Starting Sign Language Recognition. Press 'q' to quit.")

    while True:
        ret, image_frame = cam_capture.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Flip frame horizontally (mirror effect)
        image_frame = cv2.flip(image_frame, 1)

        # Crop the region of interest (ROI)
        roi = crop_image(image_frame, x, y, w, h)

        # Preprocess the ROI image
        processed_image = keras_process_image(roi)

        # Predict the letter
        confidence, pred_class = keras_predict(model, processed_image)
        predicted_letter = classes[pred_class]

        # Apply confidence and stability logic
        if confidence > CONFIDENCE_THRESHOLD and predicted_letter != 'nothing':
            if predicted_letter == last_letter:
                count_same_letter += 1
            else:
                count_same_letter = 1
                last_letter = predicted_letter

            if count_same_letter == STABILITY_THRESHOLD:
                if predicted_letter == 'space':
                    sentence += ' '
                elif predicted_letter == 'del':
                    sentence = sentence[:-1]
                else:
                    sentence += predicted_letter
                count_same_letter = 0  # Reset after adding

        # Show predicted letter and confidence
        cv2.putText(image_frame, f'Letter: {predicted_letter} ({confidence:.2f})', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Show the sentence
        cv2.putText(image_frame, f'Sentence: {sentence}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the ROI box
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

        # Display the frame
        cv2.imshow("Sign Language Recognition", image_frame)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()