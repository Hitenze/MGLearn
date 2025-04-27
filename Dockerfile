FROM ubuntu:22.04

# Install essential packages
RUN apt-get update && apt-get install -y \
   wget \
   bzip2 \
   ca-certificates \
   git \
   && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
   bash ~/miniconda.sh -b -p /opt/conda && \
   rm ~/miniconda.sh

# Add conda to path
ENV PATH /opt/conda/bin:$PATH

# Initialize conda for bash and create environment
RUN conda init bash && \
    echo "conda activate mglearn" >> ~/.bashrc

# Create a working directory
WORKDIR /app

# Copy environment.yml file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "mglearn", "/bin/bash", "-c"]

# Set up entrypoint to activate conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mglearn"]

# Default command
CMD ["/bin/bash"] 