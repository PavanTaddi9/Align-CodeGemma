FROM docker.io/elleven11/multipl-e-evaluation:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing && apt install -y --no-install-recommends \
    cargo \
    libssl-dev \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /env

RUN /env/bin/pip install --no-cache-dir --upgrade pip && \
    /env/bin/pip install --no-cache-dir \
    coverage \
    pandas \
    torch \
    jax \
    jaxlib

ENV VIRTUAL_ENV=/env
ENV PATH="/env/bin:$PATH"

COPY . /app
WORKDIR /app

RUN cargo build --release

EXPOSE 8000

ENTRYPOINT ["cargo", "run", "--release"]