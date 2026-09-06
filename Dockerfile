FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Run both CLI and web via same image: default is web
CMD ["uvicorn", "syndata.interface.app:app", "--host", "0.0.0.0", "--port", "8000"]
