# Imagen base liviana con Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo los archivos necesarios primero para evitar recacheos
COPY requirements.txt .

# Instala las dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia el resto del código al contenedor
COPY . .

# Expone el puerto de Streamlit
EXPOSE 8501

# Comando por defecto al iniciar el contenedor
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.enableCORS=false"]
