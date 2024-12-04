# Gunakan image Python
FROM python:3.9-slim

# Set direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Instal dependensi untuk aplikasi
RUN pip install --no-cache-dir -r requirements.txt

# Download model Stanza untuk bahasa Indonesia
RUN python -c "import stanza; stanza.download('id')"

# Ekspos port untuk aplikasi Flask
EXPOSE 3000

# Define the entrypoint command for the Flask application
CMD ["python", "apiflask.py"]