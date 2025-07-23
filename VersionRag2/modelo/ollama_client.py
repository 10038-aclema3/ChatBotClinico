# ollama_client.py
import requests

def generar_respuesta_llm(pregunta):
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": "llama3:8b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente clínico especializado en dermatología. "
                    "Responde con base en evidencia médica. "
                    "Incluye el código CIE-10 más probable, el tratamiento sugerido y contraindicaciones si aplica. "
                    "No uses saludos ni introducciones."
                )
            },
            {
                "role": "user",
                "content": pregunta
            }
        ]
    }
    response = requests.post(url, json=payload)
    return response.json()["choices"][0]["message"]["content"]
