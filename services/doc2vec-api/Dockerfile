
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

FROM docker.io/python:3.12.9-slim-bookworm

ADD requirements.txt /usr/src/doc2vecapi/
ADD main.py /usr/src/doc2vecapi/
WORKDIR /usr/src/doc2vecapi/

ENV PATH="$PATH:/root/.local/bin"
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8080 --workers=4"

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "gunicorn", "main:app" ]
