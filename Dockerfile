FROM python:3.10

WORKDIR .

COPY requirements.txt .
COPY model_titanic.joblib .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api_titanic.py", "--host", "0.0.0.0", "--port", "80"]