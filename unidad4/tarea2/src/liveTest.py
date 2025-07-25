import cv2
import torch
import numpy as np
import os
import yaml
from torchvision import transforms
from model import ModelCNN

class EmotionDetector:
    def __init__(self, emotion_model="../models/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_model = self.load_emotion_model(emotion_model)
        
        # Cargamos los nombres de las clases desde data.yaml
        with open('../data/data.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.labels = config['names']
        
        # Mapeamos las emociones a español
        self.spanish_labels = {
            'Anger': 'Enojado',
            'Contempt': 'Molesto',
            'Disgust': 'Asco',
            'Fear': 'Miedo',
            'Happy': 'Feliz',
            'Neutral': 'Neutral',
            'Sad': 'Triste',
            'Surprise': 'Sorpresa'
        }
        
        # TRansformaciones de imagen
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Cargar el clasificador Haar para detección de rostros
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Creamos el objeto CLAHE para mejorar el contraste
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        self.debug = False
    
    def load_emotion_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        model = ModelCNN(num_classes=8)
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
        # Aplicamos CLAHE para mejorar el contraste
        face_img = self.clahe.apply(face_img)
        return face_img
    
    def detect_emotion(self, frame):
        # Espejamos la imagen para evitar que se vea al revés
        frame = cv2.flip(frame, 1)
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
                
            # Convertimos a escala de grises
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Preprocesamos la cara
            try:
                # Apliamos preprocesamiento
                processed_face = self.preprocess_face(gray_face)
                
                # Aplicamos transformaciones
                input_tensor = self.transform(processed_face).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.emotion_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, pred = torch.max(outputs, 1)
                    emotion = self.labels[pred.item()]
                    confidence = probabilities[0][pred.item()].item()
                
                # PImprimimos probabilidades si estamos en modo debug
                if self.debug:
                    print("\nPrediction probabilities:")
                    probs = probabilities[0].cpu().numpy()
                    for i, prob in enumerate(probs):
                        print(f"{self.labels[i]}: {prob:.4f}")
                
                # NOTA IMPORTANTE: Ajustamos el umbral de confianza
                if confidence < 0.5: 
                    emotion = "Neutral"
                    confidence = 0.99
                
                # oBTEnemos la etiqueta en español
                emotion_spanish = self.spanish_labels.get(emotion, emotion)
                
                # Dibujamos el rectángulo y la etiqueta
                if emotion == "Happy":
                    color = (0, 255, 0)  # Verde para feliz
                elif emotion == "Surprise":
                    color = (0, 255, 255)  # Amarillo para sorpresa
                elif emotion in ["Anger", "Contempt", "Disgust"]:
                    color = (0, 0, 255)  # Rojo para enojo, desprecio o asco
                else:  # Fear, Neutral, Sad
                    color = (255, 0, 0)  # Azul para miedo, neutral o tristeza
                    
                # dIBUjamos el rectángulo alrededor del rostro
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Dibujamos la etiqueta con la emoción y la confianza
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
            
        # Setteamos la resolución de la cámara 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting emotion detection. Press 'q' to quit...")
        print("Press 'd' to toggle debug mode")
        
        # Creamos una ventana para mostrar la detección de emociones
        cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Emotion Detection', 800, 600)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Detectamos emociones en el frame
            result_frame = self.detect_emotion(frame)
            cv2.imshow('Emotion Detection', result_frame)
            
            # Manejamos las teclas
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