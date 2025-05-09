# Local Scripts

This directory contains utility scripts for setting up and managing your Python environment on the cluster.

## cluster_install.sh

### Purpose

The `cluster_install.sh` script automates the creation and setup of a Python virtual environment (`fdp-env`) under the `/tmp` directory on your cluster node. It installs essential tools such as JupyterLab and registers the environment as a Jupyter kernel.

### Why Install Under `/tmp`?

On many HPC clusters, the `/tmp` directory is located on local storage for each compute node. This means:
- **Faster Storage Access:** Installing and running your virtual environment from `/tmp` can be significantly faster than using your home directory or network-mounted storage, especially for I/O-intensive tasks like package imports and Jupyter operations.
- **Reduced Network Load:** Using local storage helps avoid overloading shared network filesystems, which can improve performance for both you and other users.

**Note:** `/tmp` is typically cleaned after your job ends or the node is rebooted, so you may need to reinstall the environment for each new session.

### Usage

```bash
chmod +x cluster_install.sh
./cluster_install.sh
```

This will:
- Create the `fdp-env` virtual environment in `/tmp` (if it does not already exist)
- Activate the environment
- Install/upgrade pip, JupyterLab, and ipykernel
- Register the kernel for Jupyter

To start Jupyter Lab after setup:
```bash
source /tmp/fdp-env/bin/activate
jupyter lab
```
```

Let me know if you want to add more details or sections!