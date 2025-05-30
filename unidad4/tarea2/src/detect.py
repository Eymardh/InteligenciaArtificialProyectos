import cv2
import torch
import numpy as np
import os
import glob
from model import ModelCNN
from torchvision import transforms
import yaml

class ImageEmotionDetector:
    def __init__(self, emotion_model="../models/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_model = self.load_emotion_model(emotion_model)
        
        # Cargar los nombres de las clases desde data.yaml
        with open('../data/data.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.labels = config['names']
        
        # Etiquetas en español
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
        
        # transformaciones de imagen
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
    
    def load_emotion_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        model = ModelCNN(num_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)
    
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=8,    # Icrementar para reducir falsos positivos
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def preprocess_face(self, face_img):
        # Aplicamos CLAHE para mejorar el contraste
        face_img = self.clahe.apply(face_img)
        return face_img
    
    def detect_emotion_in_image(self, image_path, output_dir="../runs/detect"):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
            
        faces = self.detect_faces(image)
        
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
                
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            try:
                # Aplicamos preprocesamiento a la cara
                processed_face = self.preprocess_face(gray_face)
                
                # Aplicamos transformaciones
                input_tensor = self.transform(processed_face).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.emotion_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, pred = torch.max(outputs, 1)
                    emotion = self.labels[pred.item()]
                    confidence = probabilities[0][pred.item()].item()
                
                # NOTA IMPORTANTE: Si la confianza es baja, asignamos "Neutral" con alta confianza
                if confidence < 0.7:
                    emotion = "Neutral"
                    confidence = 0.99
                
                # Obtener la etiqueta en español
                emotion_spanish = self.spanish_labels.get(emotion, emotion)
                
                if emotion == "Happy":
                    color = (0, 255, 0)
                elif emotion == "Surprise":
                    color = (0, 255, 255)
                elif emotion in ["Anger", "Contempt", "Disgust"]:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                    
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                label = f"{emotion_spanish} ({confidence:.2f})"
                cv2.putText(image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Guardar el resultado
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")

    def detect_emotion_in_directory(self, input_dir, output_dir="../runs/detect"):
        image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                     glob.glob(os.path.join(input_dir, "*.png")) + \
                     glob.glob(os.path.join(input_dir, "*.jpeg"))
        
        print(f"Found {len(image_paths)} images in {input_dir}")
        
        for img_path in image_paths:
            self.detect_emotion_in_image(img_path, output_dir)

if __name__ == "__main__":
    detector = ImageEmotionDetector()
    
    # NOTA PARA DETECTAR EN UNA IMAGEN INDIVIDUAL
    # detector.detect_emotion_in_image("path/to/image.jpg")
    
    # DETECTAR EMOCIONES EN UN DIRECTORIO DE IMÁGENES
    detector.detect_emotion_in_directory("../data/valid/images")