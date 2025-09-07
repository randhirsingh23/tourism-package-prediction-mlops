
FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

ARG USER=appuser
ARG UID=1000
ARG GID=1000
RUN addgroup --gid $GID $USER && \
    adduser --disabled-password --gecos "" --uid $UID --gid $GID $USER

COPY requirements.txt ./requirements.txt
COPY app.py ./app.py
RUN pip3 install --no-cache-dir -r requirements.txt

ENV HOME=/home/$USER
ENV XDG_CACHE_HOME=/home/$USER/.cache
RUN mkdir -p /home/$USER/.streamlit $XDG_CACHE_HOME && \
    printf "[server]\nheadless=true\naddress=\"0.0.0.0\"\n\n[browser]\ngatherUsageStats=false\n" \
      > /home/$USER/.streamlit/config.toml && \
    chown -R $USER:$USER /home/$USER /app

USER $USER

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
