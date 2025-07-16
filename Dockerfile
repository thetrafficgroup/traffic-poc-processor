FROM python:3.11.1-slim

WORKDIR /app

# Copy and install requirements
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full source code
COPY src/ src/

# Set the working directory to src so handler can resolve imports
WORKDIR /app/src

# Command to run when the container starts
CMD ["python", "-u", "handler.py"]