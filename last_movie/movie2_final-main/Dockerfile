# Use the official Python image from Docker Hub
FROM python:3.9

# Set environment variable to avoid writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1

# Set environment variable to ensure output is sent straight to the terminal (stdout/stderr)
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt first to leverage Docker cache and install dependencies
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the application files into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "movie_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
