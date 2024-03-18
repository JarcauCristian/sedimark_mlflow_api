FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        make \
        gcc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]
