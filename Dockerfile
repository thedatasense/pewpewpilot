FROM rust:1-bookworm AS builder

WORKDIR /build
COPY Cargo.toml build.rs ./
COPY src/ ./src/
RUN cargo build --release

FROM python:3.12-slim-bookworm

WORKDIR /app
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DURLAR_READ_ONLY=1 \
    DURLAR_POINT_STRIDE=4

# Keep huggingface_hub available for `railway ssh -- python3 /app/download_durlar.py ...`
RUN pip install --no-cache-dir huggingface_hub

COPY --from=builder /build/target/release/durlar-server /app/durlar-server
COPY download_durlar.py ./
COPY static/ ./static/
COPY calibration/ ./calibration/

CMD ["/app/durlar-server"]
