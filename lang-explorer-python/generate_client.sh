#!/bin/bash

podman run --network=host \
  -v ./src/utils/client:/local docker.io/openapitools/openapi-generator-cli:latest generate \
  -i http://localhost:8080/swagger.json \
  -g python \
  -o /local
