FROM python:3.11.1-slim

WORKDIR /

# Copy and install requirements
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary code from src to root
COPY src/ . 

CMD ["python", "-u", "/handler.py"]
