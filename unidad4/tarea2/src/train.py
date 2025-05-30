import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
import os
import numpy as np
import yaml
from collections import Counter
from model import ModelCNN
from utils import YOLOEmotionDataset, get_transforms
import matplotlib.pyplot as plt

# Cargar configuración desde data.yaml
with open('../data/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funcion para visualizar la distribución de clases
# Esta función toma los conteos de clases y los nombres de las clases, y genera un gráfico de barras
# el cual es util para entender el balance de clases en el conjunto de entrenamiento.
def plot_class_distribution(class_counts, class_names):
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts)
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Emotion Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../reports/class_distribution.png')
    plt.close()

def main():
    # Creamos el dataset y las transformaciones
    train_transform, val_transform = get_transforms()
    
    train_dataset = YOLOEmotionDataset(
        root_dir='../data',
        split='train',
        transform=train_transform
    )
    
    val_dataset = YOLOEmotionDataset(
        root_dir='../data',
        split='valid',
        transform=val_transform
    )
    
    # Calcular la distribución de clases en el conjunto de entrenamiento
    all_labels = []
    for i in range(len(train_dataset)):
        try:
            _, label = train_dataset[i]
            all_labels.append(label)
        except Exception as e:
            print(f"Skipping invalid sample: {e}")
    
    class_counts = Counter(all_labels)
    print("Class distribution in training set:")
    for i, name in enumerate(config['names']):
        count = class_counts.get(i, 0)
        print(f"  {name}: {count} samples")
    
    # Visualizar la distribución de clases
    plot_class_distribution([class_counts.get(i, 0) for i in range(len(config['names']))], 
                           config['names'])
    
    # Calculamos los pesos de clase para el muestreo ponderado
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (class_counts[i] + 1e-5) for i in range(len(config['names']))]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Cramos un muestreo ponderado para manejar el desbalance de clases
    sample_weights = [class_weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
# Creamos los DataLoaders 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Reducimos de 64 a 32
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,  # Reducimos de 64 a 32
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Inicializamos el modelo
    model = ModelCNN(num_classes=8).to(device)
    
    # Verificamos la arquitectura del modelo e imprimimos
    print("\nModel architecture:")
    print(model)
    
   # Definimos la función de pérdida con pesos de clase
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Usamos el optimizador Adam con un learning rate reducido
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Reduced LR
    
    # Learning rate scheduler para reducir el LR en caso de no mejora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=2
    )
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Creamos el directorio para guardar los modelos
    os.makedirs('../models', exist_ok=True)
    
    # Configuramos early stopping
    patience = 5
    early_stop_counter = 0
    
    # Entrenamos el modelo en un bucle
    for epoch in range(50):  # Max 50 epochs para evitar overfitting
        #  Fase de entrenamiento
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.sampler)
        train_losses.append(train_loss)
        
        # Validamos el modelo
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        # Actualizamos el scheduler
        scheduler.step(val_acc)
        
        # Imprimir el learning rate actual
        current_lr = optimizer.param_groups[0]['lr']
        
        # Imprimir resultados del epoch y metricas
        print(f'\nEpoch {epoch+1}/50')
        print(f'LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print(classification_report(
            all_labels, all_preds, 
            target_names=config['names'],
            zero_division=0
        ))
        
        # GUardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            model_path = '../models/best_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved at {model_path} with val acc: {val_acc:.4f}')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    #  guardar el modelo final
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    # Guardar las curvas de entrenamiento
    # las cuales sirven para visualizar el progreso del entrenamiento y la validación.
    plt.savefig('../reports/training_curves.png')
    plt.close()
    
    print("\nTraining completed")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    os.makedirs('../reports', exist_ok=True)
    main()