# Use a lightweight base image with Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files into the container's /app folder
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
