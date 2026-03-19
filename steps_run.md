SETUP DOCKER BY:
docker run -d -p 6333:6333 -p 6334:6334     -v $(pwd)/qdrant_storage:/qdrant/storage     qdrant/qdrant

THEN SERVE OLLAMA
ollama serve

ACTIVATE VENV
source venv/bin/activate

START BACKEND
navigate to /backend and then run:
uvicorn app.main:app --reload