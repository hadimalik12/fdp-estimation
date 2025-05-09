#!/bin/bash

set -e  # Exit on error

# Set working directory
cd /tmp

# Set up a user-specific R installation path under /tmp
R_PREFIX="/tmp/R_local_$USER"

# Download and install R locally if not already present
if [ ! -x "$R_PREFIX/bin/R" ]; then
    echo "Installing R locally in $R_PREFIX..."
    wget https://cran.r-project.org/src/base/R-4/R-4.1.2.tar.gz
    tar -xzf R-4.1.2.tar.gz
    cd R-4.1.2
    ./configure --prefix="$R_PREFIX" --enable-R-shlib
    make -j 4
    make install
    cd ..
    rm -rf R-4.1.2 R-4.1.2.tar.gz
else
    echo "Local R already installed in $R_PREFIX"
fi

# Set environment variables for local R
export PATH="$R_PREFIX/bin:$PATH"
export R_HOME="$R_PREFIX/lib64/R"

# Install fdrtool into the default library of your local R
"$R_PREFIX/bin/Rscript" -e "install.packages('fdrtool', repos='https://cloud.r-project.org')"

echo "fdrtool package installation complete in the default library of local R at $R_PREFIX."

# Check if fdp-env already exists
if [ -d "fdp-env" ]; then
    echo "Virtual environment 'fdp-env' already exists in /tmp."
else
    echo "Creating virtual environment 'fdp-env' in /tmp..."
    python3.11 -m venv fdp-env
fi

# Activate the virtual environment
source fdp-env/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
pip install jupyterlab ipykernel

# Register the kernel
python -m ipykernel install --user --name=fdp-env --display-name "Python (fdp-env)"

# Set working directory
cd ~/Desktop/fdp-estimation
# Install project dependencies from requirements.txt
pip install --upgrade -r requirements.txt