# Docker Environment for MGLearn

This directory contains Docker configuration files for the MGLearn project.

## Usage

### Building the Docker image

```bash
# From the project root
docker-compose build
```

### Running the container

```bash
# From the project root
docker-compose up -d
```

### Accessing the container

```bash
docker-compose exec mglearn bash
```

### Running Jupyter Notebook

Inside the container:
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then access Jupyter in your browser using the URL provided.

### Stopping the container

```bash
docker-compose down
```