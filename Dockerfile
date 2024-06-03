# Usa una imagen base oficial de Python 3.11 slim
FROM python:3.11-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo requirements.txt primero para aprovechar la caché de Docker layers
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de tu aplicación
COPY . .

# Informa a Docker que el contenedor escucha en el puerto 8000
EXPOSE 8000

# Ejecuta el servidor Uvicorn con live reload habilitado para desarrollo
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]