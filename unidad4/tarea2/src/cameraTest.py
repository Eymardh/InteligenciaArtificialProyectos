import cv2
import torch
import numpy as np
import os
import yaml
from torchvision import transforms
from model import EmotionCNN

class EmotionDetector:
    def __init__(self, emotion_model="../models/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_model = self.load_emotion_model(emotion_model)
        
        # Load class names from data.yaml
        with open('../data/data.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.labels = config['names']
        
        # Spanish emotion labels
        self.spanish_labels = {
            'Anger': 'Enojado',
            'Contempt': 'Desprecio',
            'Disgust': 'Asco',
            'Fear': 'Miedo',
            'Happy': 'Feliz',
            'Neutral': 'Neutral',
            'Sad': 'Triste',
            'Surprise': 'Sorpresa'
        }
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create CLAHE object for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Debug mode
        self.debug = False
    
    def load_emotion_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        model = EmotionCNN(num_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def preprocess_face(self, face_img):
        # Apply CLAHE for better contrast normalization
        face_img = self.clahe.apply(face_img)
        return face_img
    
    def detect_emotion(self, frame):
        # Mirror frame for more natural UX
        frame = cv2.flip(frame, 1)
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
                
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Preprocess and predict
            try:
                # Apply preprocessing
                processed_face = self.preprocess_face(gray_face)
                
                # Apply transformations
                input_tensor = self.transform(processed_face).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.emotion_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, pred = torch.max(outputs, 1)
                    emotion = self.labels[pred.item()]
                    confidence = probabilities[0][pred.item()].item()
                
                # Print probabilities in debug mode
                if self.debug:
                    print("\nPrediction probabilities:")
                    probs = probabilities[0].cpu().numpy()
                    for i, prob in enumerate(probs):
                        print(f"{self.labels[i]}: {prob:.4f}")
                
                # Confidence thresholding
                if confidence < 0.6:  # Lowered threshold
                    emotion = "Neutral"
                    confidence = 0.99
                
                # Get Spanish label
                emotion_spanish = self.spanish_labels.get(emotion, emotion)
                
                # Draw results with emotion-dependent colors
                if emotion == "Happy":
                    color = (0, 255, 0)  # Green
                elif emotion == "Surprise":
                    color = (0, 255, 255)  # Yellow
                elif emotion in ["Anger", "Contempt", "Disgust"]:
                    color = (0, 0, 255)  # Red
                else:  # Fear, Neutral, Sad
                    color = (255, 0, 0)  # Blue
                    
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label with background for readability
                label = f"{emotion_spanish} ({confidence:.2f})"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y-35), (x+text_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        # Set higher resolution for better face detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting emotion detection. Press 'q' to quit...")
        print("Press 'd' to toggle debug mode")
        
        # Create window with fixed size
        cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Emotion Detection', 800, 600)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Detect and display emotions
            result_frame = self.detect_emotion(frame)
            cv2.imshow('Emotion Detection', result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('d'):
                self.debug = not self.debug
                print(f"Debug mode {'ON' if self.debug else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run()