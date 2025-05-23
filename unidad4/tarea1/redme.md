# Tarea 1 Unidad 4 Dataset Imagenes

El dataset se obtuvo del siguiente [enlace](https://www.kaggle.com/datasets/msambare/fer2013)

lo descargamos usando el comando CURL

```bash
curl -L -o fer2013.zip https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013
```

una vez obtenidas las imagenes las descomprimimos

```bash
unzip fer2013.zip -d ./unidad4/tarea1
```

después mejoramos el brillo de las imagenes usando ImageMagick

```bash
find . -type f -iname '*.jpg' -exec convert '{}' -modulate 120,100,100 '{}' \;
```
