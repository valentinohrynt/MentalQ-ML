# Gunakan image Python
FROM python:3.9-slim

# Set direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Instal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Ekspos port untuk aplikasi Flask
EXPOSE 3000

# Perintah untuk menjalankan aplikasi
CMD ["python", "apiflask.py"]