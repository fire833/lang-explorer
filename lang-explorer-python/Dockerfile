
# Copyright (C) 2026 Kendall Tauser
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

FROM docker.io/python:3.12.9-slim-bookworm

ADD . /usr/src/lang-explorer/
WORKDIR /usr/src/lang-explorer/

ENV PATH="$PATH:/root/.local/bin"

RUN wget -O /usr/local/bin/dumb-init https://static.tauser.us/init/dumb-init-x86 && chmod +x /usr/local/bin/dumb-init

RUN pip install pipx && pipx install poetry && poetry install --no-root

ENTRYPOINT [ "dumb-init", "./main.py" ]
