import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

# --- CONFIGURACIÓN ---
ARCHIVO_CSV = os.path.join("data", "SINTOMAS_CIE10.csv")
RUTA_CHROMA = "./chroma_db"
COLLECTION_NAME = "casos_derma_ext"

# --- CARGA DEL CSV ---
print("📥 Cargando datos desde el archivo CSV...")
df = pd.read_csv(ARCHIVO_CSV, sep=";", encoding="latin1")

# --- VALIDACIÓN DE COLUMNAS NECESARIAS ---
columnas_necesarias = {
    "motivo_consulta", "cie_10", "tratamiento_recomendado",
    "contraindicaciones", "nombre_cie_10",
    "zona_afectada", "edad_aproximada", "sexo",
    "frecuencia_episodios", "intensidad"
}
if not columnas_necesarias.issubset(df.columns):
    faltantes = columnas_necesarias - set(df.columns)
    raise ValueError(f"Faltan columnas en el CSV: {faltantes}")

# --- CARGA DEL MODELO DE EMBEDDINGS ---
print("🧠 Generando embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = df["motivo_consulta"].tolist()
embeddings = model.encode(texts).tolist()

# --- INICIALIZAR CHROMA CLIENT ---
client = chromadb.PersistentClient(path=RUTA_CHROMA)

# --- ELIMINAR Y CREAR COLECCIÓN ---
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"🗑️ Colección '{COLLECTION_NAME}' eliminada.")
except:
    print(f"ℹ️ Colección '{COLLECTION_NAME}' no existía.")

collection = client.create_collection(name=COLLECTION_NAME)
print(f"✅ Nueva colección '{COLLECTION_NAME}' creada.")

# --- INSERTAR DATOS EN CHROMA ---
print("📌 Insertando registros en ChromaDB...")
for i, row in df.iterrows():
    texto = row["motivo_consulta"]
    embedding = embeddings[i]

    metadata = {
        "cie_10": row["cie_10"],
        "nombre_cie_10": row["nombre_cie_10"],
        "tratamiento_recomendado": row["tratamiento_recomendado"],
        "contraindicaciones": row["contraindicaciones"],
        "zona_afectada": row["zona_afectada"],
        "edad_aproximada": row["edad_aproximada"],
        "sexo": row["sexo"],
        "frecuencia_episodios": row["frecuencia_episodios"],
        "intensidad": row["intensidad"]
    }

    collection.add(
        documents=[texto],
        ids=[f"caso_{i}"],
        embeddings=[embedding],
        metadatas=[metadata]
    )

print(f"✅ {len(df)} documentos cargados exitosamente en ChromaDB.")
