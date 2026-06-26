#!/bin/bash

# reset the data directory
rm -rf data && mkdir data

# download data from kaggle and clean up unnecessary files
cd data
uv run kaggle datasets download retailrocket/ecommerce-dataset
unzip ecommerce-dataset.zip
rm -f ecommerce-dataset.zip
