
# Copyright (C) 2025 Kendall Tauser
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

FROM docker.io/rust:latest AS builder

ADD src /usr/src/lang-explorer/src
ADD Cargo.lock /usr/src/lang-explorer
ADD Cargo.toml /usr/src/lang-explorer
WORKDIR /usr/src/lang-explorer

RUN wget -O /usr/local/bin/dumb-init https://static.tauser.us/init/dumb-init-x86 && chmod +x /usr/local/bin/dumb-init

# Compile for musl so it works on alpine: https://stackoverflow.com/questions/59766239/how-to-build-a-rust-app-free-of-shared-libraries#59766875
RUN RUSTFLAGS='-C link-arg=-s' cargo build --release --bin explorer

FROM docker.io/ubuntu:noble

COPY --from=builder /usr/local/bin/dumb-init /usr/local/bin/dumb-init
COPY --from=builder /usr/src/lang-explorer/target/release/explorer /usr/local/bin/explorer
USER 1000:1000

LABEL org.opencontainers.image.author="Kendall Tauser"
LABEL org.opencontainers.image.title="Language Explorer API."
# LABEL org.opencontainers.image.source="https://github.com/fire833/lang-explorer"

ENTRYPOINT [ "dumb-init", "explorer" ]
