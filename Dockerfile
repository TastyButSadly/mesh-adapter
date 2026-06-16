FROM firedrakeproject/firedrake-vanilla-default:latest

# Install system deps for gmsh
RUN apt-get update -qq && apt-get install -y -qq libgl1 libglu1-mesa > /dev/null 2>&1 && rm -rf /var/lib/apt/lists/*

# Install adapter dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir gmsh numpy scipy matplotlib pillow meshio

# Copy project code
COPY src/ /work/src/
COPY scripts/ /work/scripts/
COPY examples/ /work/examples/
COPY pyproject.toml /work/
COPY docker-entrypoint.py /work/

# Install diff-mesh-adapter
RUN cd /work && pip install --no-cache-dir -e .

WORKDIR /work

ENTRYPOINT ["python3", "docker-entrypoint.py"]
