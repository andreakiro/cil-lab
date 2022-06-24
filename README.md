# Computational Intelligence Lab - FS2022

- [Resources](https://docs.google.com/document/d/1ynT7xilJTBtD7T8KpMyKRjc3CC-wEh-XX7ZgRkh4fyc/edit#heading=h.ajjlw0b7sp4p)

## How to install and run the environment
Verify that conda is installed and running on your system by typing:
```
conda --version
```
To install the environment, simply run the following commands in the main project folder:
```
conda env create
conda activate cil
```

## How to submit to Kaggle leaderboard
Install Kaggle API using the command:
```
pip install kaggle
```
To use the Kaggle API, sign up for a [Kaggle](https://www.kaggle.com) account. Then go to the 'Account' tab of your user profile and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json`.
For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command:
```
chmod 600 ~/.kaggle/kaggle.json
```
Once the token is set up, just use the in-line command:
```
kaggle competitions submit -c cil-collaborative-filtering-2022 -f submission.zip -m "message"
```
to submit the file `submission.zip` to the competition leaderboard.

## ETH Euler Instructions
To **transfer the project folder to the cluster** without pushing it to GitHub, use
```
scp -r ../cil-lab nethz@euler.ethz.ch:~/
```
To **set up the environment** in the cluster, use
```
# connect to cluster
ssh creds@euler.ethz.ch

# install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh
rm miniconda.sh

# disable base env.
conda config --set auto_activate_base false

# move to the project folder
cd ~/cil-lab/

# create and activate the environment
conda env create
conda activate cil
```
To **submit a CPU** job to the cluster, use
```
ssh nethz@euler.ethz.ch
env2lmod
module load gcc/6.3.0 eth_proxy hdf5/1.10.1
cd ~/cil-lab/
conda activate cil
bsub -W HH:MM -n numberofcpus -R "rusage[mem=8192]" python main.py 
```
Other useful cluster commands:
- `busers`	user limits, number of pending and running jobs
- `bqueues`	queues status (open/closed; active/inactive)
- `bjobs`	more or less detailed information about pending, running and recently finished jobs
- `bbjobs`	better bjobs (bjobs with human readable output)
- `bhist`	information about jobs that finished in the last hours/days
- `bpeek`	display the standard output of a given job
- `lsf_load`	show the CPU load of all nodes used by a job
- `bjob_connect`	login to a node where one of your jobs is running
- `bkill`	kill a job
