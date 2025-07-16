FROM python:3.11.1-slim

WORKDIR /

# Copy and install requirements
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code
COPY . .

# Command to run when the container starts
CMD ["python", "-u", "/src/handler.py"]