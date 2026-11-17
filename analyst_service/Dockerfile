FROM python:3.11-slim
WORKDIR /app
COPY analyst_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY analyst_service ./analyst_service
RUN useradd -m analyst && mkdir /data && chown analyst:analyst /data
USER analyst
ENV ANALYST_DATA_DIR=/data OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
EXPOSE 8003
CMD ["uvicorn","analyst_service.app:app","--host","0.0.0.0","--port","8003"]
