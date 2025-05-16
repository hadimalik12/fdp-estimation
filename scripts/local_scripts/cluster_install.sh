#!/bin/bash

set -e  # Exit on error

# Set installation prefix
R_PREFIX="/tmp/R_local_wei402"

# Create installation directory if it doesn't exist
mkdir -p "$R_PREFIX"

# Use system curl and libraries
export PATH="/usr/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64:$LD_LIBRARY_PATH"
export CPATH="/usr/include:$CPATH"

module load gcc/14.2.0
module load openblas/0.3.27

# Download and install R locally if not already present
if [ ! -x "$R_PREFIX/bin/R" ]; then
    echo "Installing R locally in $R_PREFIX..."
    if [ ! -f "R-4.1.2.tar.gz" ]; then
        echo "Downloading R-4.1.2..."
        wget https://cran.r-project.org/src/base/R-4/R-4.1.2.tar.gz
    fi
    tar -xzf R-4.1.2.tar.gz
    cd R-4.1.2
    ./configure --prefix="$R_PREFIX" \
        --enable-R-shlib \
        --with-x=no \
        --with-blas="-lopenblas" \
        --with-lapack="-lopenblas" \
        LDFLAGS="-L/apps/spack/bell-20250305/apps/openblas/0.3.27/lib" \
        CPPFLAGS="-I/apps/spack/bell-20250305/apps/openblas/0.3.27/include"
    make -j4
    make install
    cd ..
    rm -rf R-4.1.2 R-4.1.2.tar.gz
else
    echo "Local R already installed in $R_PREFIX"
fi

# Install required R packages
"$R_PREFIX/bin/R" -e "install.packages('fdrtool', repos='https://cloud.r-project.org')"

# Check if fdp-env already exists
if [ -d "/tmp/fdp-env" ]; then
    echo "Virtual environment 'fdp-env' already exists in /tmp."
else
    echo "Creating virtual environment 'fdp-env' in /tmp..."
    python3.8 -m venv /tmp/fdp-env
fi

# Activate the virtual environment
source /tmp/fdp-env/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
# Install rpy2
pip install rpy2

echo "R and rpy2 installation completed successfully!"

pip install jupyterlab ipykernel

# # Register the kernel
# python -m ipykernel install --user --name=fdp-env --display-name "Python (fdp-env)"

# --- Custom Jupyter kernel with OpenBLAS and R env ---

# Create the custom kernel spec directory
KERNEL_DIR=~/.local/share/jupyter/kernels/fdp-env-custom
mkdir -p "$KERNEL_DIR"

# Write the kernel.json
cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "$(pwd)/scripts/local_scripts/r-bell-kernel-install.sh",
    "/tmp/fdp-env/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (fdp-env with OpenBLAS)",
  "language": "python"
}
EOL

# Set working directory
cd ~/Desktop/fdp-estimation/scripts/local_scripts
# Install project dependencies from requirements.txt
pip install --upgrade -r requirements.txt