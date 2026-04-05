FROM python:3.10-slim

# Create a non-root user for security (required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . .

# Expose exactly port 7860 as requested by HuggingFace Spaces
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
