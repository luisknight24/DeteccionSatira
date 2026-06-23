# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000 for Hugging Face Spaces security compliance
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user gunicorn

# Pre-download NLTK resources and spaCy model during build time
RUN python -m nltk.downloader punkt punkt_tab stopwords vader_lexicon
RUN python -m spacy download es_core_news_sm

# Copy the application source code
COPY --chown=user . .

# Expose the default port (7860 is required by Hugging Face Spaces)
EXPOSE 7860

# Run Django with gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "180", "satire_detector_api.wsgi:application"]
