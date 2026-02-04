# Use Python 3.10 to match your Azure env
FROM python:3.10-slim

# --- Create non-root user -----------------------------------------------------
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR $HOME/app

# --- Copy requirements first for caching -------------------------------------
COPY requirements.txt ./

# --- Install OS dependencies --------------------------------------------------
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      git \
      build-essential \
  && rm -rf /var/lib/apt/lists/*

# --- Install Python dependencies ---------------------------------------------
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt \
 && python -m streamlit --version

# --- Copy application code ----------------------------------------------------
COPY --chown=user . $HOME/app

# --- Switch to non-root user --------------------------------------------------
USER user

# --- Configure Streamlit server ----------------------------------------------
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# --- Launch the Streamlit app -------------------------------------------------
ENTRYPOINT ["python", "-m", "streamlit", "run", "src/streamlit_app.py",  "--server.port=8501",  "--server.address=0.0.0.0",  "--server.headless=true",  "--server.enableXsrfProtection=false"]