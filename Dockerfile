# Use a lightweight Python base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker's caching for dependency installation
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt in a dedicated folder to ensure it's available
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt

# Copy the remaining application files
COPY . .

# Expose the Flask application port
EXPOSE 3000

# Define the entrypoint command for the Flask application
CMD ["python", "apiflask.py"]
