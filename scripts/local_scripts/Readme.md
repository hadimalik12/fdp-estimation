# Local Scripts 
---
This guide provides instructions for setting up your development environment on an HPC cluster. It shows how to configure the workspace for optimal performance using local storage and enable seamless interactive development through VS Code (Cursor).

---

## Environment Setup in `/tmp`

In many HPC clusters, the `/tmp` directory is local to each compute node, providing significantly faster storage access compared to home directories or network-mounted storage. Setting up your Python environment (e.g., `fdp-env`) in `/tmp` offers several advantages:
- **Enhanced package import speed and I/O operations**
- **Decreased load on shared filesystems**
- **Improved performance for Jupyter Lab and other interactive tools**

**Important:** The `/tmp` directory is usually cleared after your job concludes or the node reboots. Therefore, you will need to reinstall the environment for each new session. If you wish to install the environment permanently, modify the script installation command as needed.

---

### Workflow: Using VS Code and Jupyter Lab on a Compute Node

#### 1. SSH to the Login Node

Connect to the cluster's login node using your SSH key:
```bash
ssh cluster-login
```
*(Replace `cluster-login` with your actual login node alias.)*

#### Keeping SLURM Sessions Alive with tmux

To maintain your cluster-computation session after disconnecting from the initial login, use tmux:

1. Create a new tmux session:
```bash
tmux new -s mysession
```

2. Detach from tmux (safe to disconnect SSH):
```bash
Ctrl + b, then d
```

3. Reconnect later from any terminal:
```bash
ssh cluster-login
tmux attach -t mysession
```

4. After completing your computation, terminate the tmux session:
```bash
# Option 1: From outside the session
ssh cluster-login
tmux ls
tmux kill-session -t mysession

# Option 2: From inside the session
exit
```

---

#### 2. Request an Interactive Compute Node

Run:
```bash
sinteractive -A your_account_name -n 12 -t 12:00:00
```
where `-n 12`: Number of CPU cores and `-t 12:00:00`: Walltime (12 hours)

---

#### 3. Find Your Compute Node Name

Check your job and node with:
```bash
squeue -u $USER
```
*(Replace $USER with your username.)*

---

#### 4. Set Up SSH Config for Two-Hop Access

Edit your `~/.ssh/config` to include:

```bash
Host login-alias
    HostName your-login-hostname
    User your-username
    IdentityFile /path/to/your/ssh_key

Host compute-alias
    HostName your-compute-hostname
    User your-username
    ProxyJump login-alias
    IdentityFile /path/to/your/ssh_key
```

#### 5. Connect to the Compute Node in VS Code

- Open a new VS Code window.
- Use the Remote-SSH extension to connect to `compute-alias`.
- **Important:** Keep your original SSH session open, as closing it may terminate your interactive job.

---

## Script Usage

The `cluster_install.sh` script in this directory automates the creation and setup of the `fdp-env` virtual environment under `/tmp` on your compute node.

**To use:**
```bash
chmod +x r-bell-kernel-install.sh
chmod +x cluster_install.sh
./cluster_install.sh
```

## Generating SGD samples

To generate SGD samples, you can use either the Python script directly or the shell script wrapper:

### Using the shell script (recommended):

1. Run with default settings (32 samples, 32 workers):
```bash
bash local_scripts/run_generate_samples.sh
```

2. Run with all parameters specified:
```bash
bash local_scripts/run_generate_samples.sh \
    --num_samples 64 \
    --num_workers 8 \
    --internal_result_path "/path/to/results" \
    --model_type "CNN"
```

### Using the Python script directly:

```bash
python scripts/sgd_experiment/generate_sgd_samples.py \
    --num_samples 32 \
    --num_workers 4 \
    --internal_result_path "/path/to/results" \
    --model_type "CNN"
```

The script will:
- Generate the specified number of samples from both distributions
- Save the full trainned SGD models in the specified path
- Save the pair of distribution observations under project_root/data folder

### Running using batch file

To run the sample generation using SLURM batch system:

1. Submit the batch job:
```bash
sbatch scripts/local_scripts/generate_sgd_samples.sub
```

2. Monitor the job status:
```bash
squeue -u $USER
```

3. Check the output and error logs in:
```
log/sbatch/sgd_samples_<jobid>.out
log/sbatch/sgd_samples_<jobid>.err
```

The batch script will:
- Run the installation script
- Generate samples using parallel workers
- Save trained models in the specified results directory
- Log detailed information including timing statistics

**important* Before running, please adjust the parameters in `generate_sgd_samples.sub` to match your requirements:

To cancel a running job:
```bash
scancel <jobid>
``` 