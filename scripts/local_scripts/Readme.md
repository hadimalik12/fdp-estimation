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

## Workflow: Using VS Code and Jupyter Lab on a Compute Node

### 1. SSH to the Login Node

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

### 2. Request an Interactive Compute Node

Run:
```bash
sinteractive -A your_account_name -n 12 -t 12:00:00
```
where `-n 12`: Number of CPU cores and `-t 12:00:00`: Walltime (12 hours)

---

### 3. Find Your Compute Node Name

Check your job and node with:
```bash
squeue -u $USER
```
*(Replace $USER with your username.)*

---

### 4. Set Up SSH Config for Two-Hop Access

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

### 5. Connect to the Compute Node in VS Code

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
