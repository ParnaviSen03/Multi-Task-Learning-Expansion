FROM python:3.9-slim

# Copy and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade --no-cache-dir pip && pip install -r /tmp/requirements.txt

# Set working directory
WORKDIR /app

# Copy notebook file
COPY ML-Apprentice-Fetch.ipynb .

# Expose Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
