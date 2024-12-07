FROM python:3.11.8-slim

WORKDIR /app

COPY ./serve-requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r serve-requirements.txt && \
    rm -rf /usr/local/lib/python3.11/site-packages/plotly* \
    /usr/local/lib/python3.11/site-packages/pyarrow* \
    /usr/local/lib/python3.11/site-packages/seaborn* \
    /usr/local/lib/python3.11/site-packages/sympy* \
    /var/lib/apt/lists/*

COPY ./model_dir /app/model_dir
COPY ./main.py /app/main.py

ENTRYPOINT ["python", "-m", "main"]
