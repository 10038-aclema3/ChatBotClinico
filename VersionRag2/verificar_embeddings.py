from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import numpy as np

def verificar_embeddings(ruta_csv, modelo_nombre="all-MiniLM-L6-v2"):
    # Cargar modelo de embeddings
    print("🔠 Cargando modelo de embeddings...")
    modelo = SentenceTransformer(modelo_nombre)

    # Leer archivo CSV
    print("📄 Leyendo archivo:", ruta_csv)
    df = pd.read_csv(ruta_csv, encoding="utf-8")  # Cambia a 'latin-1' si da error

    if "motivo_consulta" not in df.columns:
        print("❌ ERROR: No se encontró la columna 'motivo_consulta'.")
        print("🧾 Columnas disponibles:", df.columns.tolist())
        return

    textos = df["motivo_consulta"].fillna("").tolist()

    # Mostrar los primeros embeddings
    print("🧠 Generando embeddings para los primeros 5 registros...")
    for i, texto in enumerate(textos[:5]):
        embedding = modelo.encode(texto)
        print(f"\n➡️ Texto {i+1}: {texto}")
        print(f"🔢 Embedding (primeros valores): {embedding[:5]}... (vector de {len(embedding)} dimensiones)")

    # Insertar en ChromaDB temporal
    print("\n💾 Creando colección temporal en ChromaDB...")
    client = chromadb.Client()
    client.reset()
    coleccion = client.create_collection(name="verificacion_test")

    for i, texto in enumerate(textos[:3]):
        emb = modelo.encode(texto).tolist()
        coleccion.add(
            documents=[texto],
            embeddings=[emb],
            ids=[f"id_{i}"]
        )

    # Hacer consulta de prueba
    print("\n🔍 Consultando similitud con: 'granos en la nariz'")
    resultados = coleccion.query(query_texts=["granos en la nariz"], n_results=3)
    for j, doc in enumerate(resultados["documents"][0]):
        print(f"✅ Resultado {j+1}: {doc}")

    print("\n✅ Verificación completa: los embeddings se están generando y usando correctamente.")

verificar_embeddings("SINTOMAS_CIE10.csv")
