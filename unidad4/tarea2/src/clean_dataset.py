import os
import glob

def clean_dataset(root_dir):
    # RElimina archivos de etiquetas vacíos y sus imágenes correspondientes
    label_files = glob.glob(os.path.join(root_dir, "*", "labels", "*.txt"))
    for lf in label_files:
        if os.path.getsize(lf) == 0:
            print(f"Removing empty label file: {lf}")
            os.remove(lf)
            
            # Elimina la imagen correspondiente
            img_file = lf.replace("labels", "images").replace(".txt", ".jpg")
            if os.path.exists(img_file):
                print(f"Removing corresponding image: {img_file}")
                os.remove(img_file)
    
    # Elimina imágenes y etiquetas con caras muy pequeñas
    for split in ["train", "valid"]:
        label_dir = os.path.join(root_dir, split, "labels")
        for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                
                line = lines[0].strip().split()
                if len(line) < 5:
                    continue
                
                # Obtener las coordenadas y dimensiones de la cara
                _, _, _, bw, bh = map(float, line[:5])
                if bw < 0.1 or bh < 0.1:  # Remove very small faces
                    img_file = label_file.replace("labels", "images").replace(".txt", ".jpg")
                    if os.path.exists(img_file):
                        print(f"Removing small face: {img_file}")
                        os.remove(img_file)
                        os.remove(label_file)

if __name__ == "__main__":
    clean_dataset("../data")