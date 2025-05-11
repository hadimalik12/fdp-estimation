#!/bin/bash
module load openblas/0.3.27
export LD_LIBRARY_PATH=/tmp/R_local_wei402/lib64/R/lib:/apps/spack/bell-20250305/apps/openblas/0.3.27/lib:$LD_LIBRARY_PATH
export R_HOME=/tmp/R_local_wei402/lib64/R
exec "$@"