FROM rust:1.82-slim-bookworm as build

RUN apt-get update
RUN apt-get install -y build-essential cmake libssl-dev pkg-config libclang-dev libvulkan-dev

COPY src /app/src
COPY README.md /app
COPY Cargo.toml /app

WORKDIR /app
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
ENV RUSTFLAGS="-C target-cpu=native"
RUN cargo build --release

RUN strip /app/target/release/rusty_llm

FROM debian:bookworm-slim

RUN apt-get update && apt install -y openssl libvulkan1 libgomp1

WORKDIR /app
COPY --from=build /app/target/release/rusty_llm /app/rusty_llm
COPY data data/

ENTRYPOINT ["/app/rusty_llm"]
