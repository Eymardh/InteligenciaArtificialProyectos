# es importande instalar las dependencias antes de iniciar
# pip install pandas scikit-learn nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("./spam_assassin.csv")
# nltk.download("stopwords") # Descomentar la primera vez
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    texto = texto.lower() 
    texto = re.sub(r'\W', ' ', texto)
    palabras = texto.split()  
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    return ' '.join(palabras)

df["text"] = df["text"].apply(limpiar_texto)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["target"]

X_train, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)

def predecir_spam(texto):
    texto_limpio = limpiar_texto(texto) 
    texto_vectorizado = vectorizer.transform([texto_limpio])  
    prediccion = modelo.predict(texto_vectorizado)[0] 
    return "Spam" if prediccion == 1 else "No Spam"

correo_prueba = "Congratulations! You've won a free iPhone. Click here to claim your prize now!"
print(predecir_spam(correo_prueba))

