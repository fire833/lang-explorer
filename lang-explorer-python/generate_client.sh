#!/bin/bash

podman run --network=host \
  -v ./src/utils/client:/local docker.io/openapitools/openapi-generator-cli:latest generate \
  -i http://localhost:8080/swagger.json \
  -g python-pydantic-v1 \
  -o /local

prefix="src/utils/client"

rm -rf $prefix/.github $prefix/docs $prefix/.openapi-generator $prefix/test
rm $prefix/*
rm $prefix/.gitignore $prefix/.travis.yml $prefix/.gitlab-ci.yml $prefix/.openapi-generator-ignore
mv $prefix/openapi_client/* $prefix
rm -rf $prefix/openapi_client

find $prefix -type f -name *.py -exec sed -i 's/from openapi_client/from src.utils.client/g' {} \;
