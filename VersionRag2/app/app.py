# --- IMPORTACIONES ---
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import os
import sys
import time

# --- IMPORTAR OLLAMA DESDE /modelo ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "modelo")))
from ollama_client import generar_respuesta_llm

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Chatbot Clínico", page_icon="🧠", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧴 Chatbot Clínico Dermatológico</h1>", unsafe_allow_html=True)

# --- CARGAR ESTILOS CSS DESDE ARCHIVO EXTERNO ---
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- CONEXIÓN A CHROMADB Y MODELO DE EMBEDDINGS ---
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "casos_derma_ext"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# --- FUNCIÓN PRINCIPAL MODIFICADA ---
def buscar_respuesta(motivo):
    embedding = embedding_model.encode(motivo).tolist()
    resultado = collection.query(
        query_embeddings=[embedding],
        n_results=5,  # Traer hasta 5 coincidencias
        include=["documents", "metadatas"]
    )

    respuestas_html = ""

    if resultado["metadatas"] and resultado["metadatas"][0]:
        for idx in range(len(resultado["metadatas"][0])):
            datos = resultado["metadatas"][0][idx]
            similitud_texto = resultado["documents"][0][idx]

            if motivo.lower() in similitud_texto.lower() or len(motivo.split()) <= 4:
                cie = datos.get("cie_10", "No encontrado")
                nombre_cie = datos.get("nombre_cie_10", "Descripción no disponible")
                tratamiento = datos.get("tratamiento_recomendado", "Sin tratamiento disponible")
                contra = datos.get("contraindicaciones", "Sin contraindicaciones registradas")

                tratamientos = [t.strip() for t in tratamiento.split(".") if t.strip()]
                contraindicaciones = [c.strip() for c in contra.split(".") if c.strip()]

                respuestas_html += f"""
<div class='chatbot'>
    <h4>🩺 <b>CIE-10:</b> {cie} – {nombre_cie}</h4>
    <p><b>💊 Tratamientos sugeridos:</b></p>
    <ul>
        {''.join(f"<li>{t}</li>" for t in tratamientos)}
    </ul>
    <p><b>⚠️ Contraindicaciones:</b></p>
    <ul>
        {''.join(f"<li>{c}</li>" for c in contraindicaciones)}
    </ul>
</div>
<br>
""".strip()

    if respuestas_html:
        return respuestas_html

    return f"<div class='chatbot'>{generar_respuesta_llm(motivo)}</div>"

# --- MENSAJE INICIAL ---
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "<i>Hola, soy tu chatbot clínico inteligente. ¿En qué puedo ayudarte hoy?</i>"}
    ]

# --- MOSTRAR HISTORIAL DE MENSAJES CON ESTILO ---
for mensaje in st.session_state.chat:
    if mensaje["role"] == "user":
        st.markdown(
            f"""
            <div class="user">
                👨‍⚕️ <b>Médico:</b><br>{mensaje["content"]}
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="bot">
                🤖 {mensaje["content"]}
            </div>
            """, unsafe_allow_html=True
        )

# --- ENTRADA DEL MÉDICO + ANIMACIÓN DE CARGA ---
pregunta = st.chat_input("Escribe el motivo de consulta...")

if pregunta:
    st.session_state.chat.append({"role": "user", "content": pregunta})

    placeholder = st.empty()
    for i in range(3):
        placeholder.markdown(f"<i>🤖 Pensando{'.' * (i+1)}</i>", unsafe_allow_html=True)
        time.sleep(0.4)
    respuesta = buscar_respuesta(pregunta)
    placeholder.empty()

    st.session_state.chat.append({"role": "assistant", "content": respuesta})
    st.rerun()
