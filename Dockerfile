FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the necessary port (if applicable)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]