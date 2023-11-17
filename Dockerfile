# Użyj obrazu Python jako bazowego obrazu
FROM python:3.11

# Ustaw katalog roboczy w kontenerze
WORKDIR /kwiatki_new

# Skopiuj pliki z obecnego katalogu do katalogu /app w kontenerze
COPY . /kwiatki_new

# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

# Otwórz port 8000
EXPOSE 8000

# Uruchom aplikację FastAPI po starcie kontenera
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
