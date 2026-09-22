FROM python:3.12-slim
WORKDIR /app
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home modeluser
USER modeluser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "intentlab.api:app", "--host", "0.0.0.0", "--port", "8000"]
