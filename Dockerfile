FROM firedrake-movement:latest

# Install adapter dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir numpy scipy matplotlib pillow meshio

# Install gmsh Python bindings (works on x86_64 Linux)
RUN pip install --no-cache-dir gmsh

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
