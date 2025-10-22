# ---------- 1️⃣  Base image ----------
FROM python:3.10-slim

# ---------- 2️⃣  Working directory ----------
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# ---------- 3️⃣  System dependencies ----------
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# ---------- 4️⃣  Python dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 5️⃣  Copy all project files ----------
COPY . .

# ---------- 6️⃣  Expose the FastAPI port ----------
EXPOSE 8000

# ---------- 7️⃣  Command to start FastAPI ----------
CMD ["uvicorn", "app.server_ollama:app", "--host", "0.0.0.0", "--port", "8000"]