FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('words', download_dir='./nltk_data')"

# Create directories
RUN mkdir -p PDFs input output && \
    chmod -R 755 PDFs input output

# Set Python to unbuffered mode
ENV PYTHONUNBUFFERED=1

# Run the script
CMD ["python", "-u", "main.py"]