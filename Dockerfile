# Gunakan image Python
FROM python:3.9-slim

# Set direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Instal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt
RUN python -m nltk.downloader punkt stopwords

# Jika Anda menggunakan Stanza, pastikan untuk mendownload model bahasa Indonesia
RUN python -c "import stanza; stanza.download('id')"

# Ekspos port untuk aplikasi Flask
EXPOSE 3000

# Perintah untuk menjalankan aplikasi
CMD ["python", "apiflask.py"]
