# Running the NT pipeline on the moriah cluster

Step-by-step instructions to clone the repo, set everything up and score the
ten annotated stories on a GPU node. Every command is preceded by a comment
explaining what it does. This is exactly the procedure of Dan's successful
run of 2026-08-14 (SLURM job 45865077, details at the bottom).

## 1. Clone the repo

```bash
# Connect to the moriah gateway (SLURM commands need a login shell there)
ssh <your-user>@moriah

# Go to your LAB storage, NOT your home directory — home has a 5 GB quota
# and the model file alone is 4.2 GB
cd /sci/labs/<your-lab>/<your-user>

# Clone the repository (code + docs; data, model and outputs are not in git)
git clone https://github.com/DanAbergel/narrative-creativity.git

# Enter the project directory — all following commands run from here
cd narrative-creativity
```

## 2. Python environment

```bash
# Create a virtual environment (moriah's system python3 is 3.11 — fine)
python3 -m venv .venv

# Upgrade pip inside the venv
.venv/bin/pip install --upgrade pip

# Install the xlsx reader/writer (the only pure-Python dependency needed
# for the llamacpp backend; torch/transformers are only for the "hf"
# debug backend and can be skipped)
.venv/bin/pip install openpyxl

# Download the CUDA build of llama-cpp-python (1.8 GB wheel). pip's own
# download from GitHub tends to die mid-way on moriah, so fetch it with
# curl and resume (-C -) until complete — re-run the loop if needed
for i in 1 2 3 4 5; do curl -L -C - -o /tmp/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl \
  "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-cu124/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl" && break; done

# Install the wheel into the venv
.venv/bin/pip install /tmp/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl

# Install the CUDA runtime libraries (libcudart, libcublas) via pip —
# the compute nodes provide the driver (libcuda) but not the runtime;
# the sbatch script points LD_LIBRARY_PATH at these
.venv/bin/pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

# Clean up the wheel
rm /tmp/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl
```

Note: `import llama_cpp` FAILS on the gateway (no GPU driver there) — that
is expected; it works on the compute nodes.

## 3. Model

```bash
# Download the quantized Hebrew-Mistral-7B (4.2 GB) straight from
# Hugging Face into models/ — this is the exact file used for all runs
curl -L -o models/Hebrew-Mistral-7B.Q4_K_M.gguf \
  "https://huggingface.co/mradermacher/Hebrew-Mistral-7B-GGUF/resolve/main/Hebrew-Mistral-7B.Q4_K_M.gguf"

# Verify the checksum — it must print the hash below, otherwise re-download
# expected: 71b041ae46aea2847624233781f450d4cf3ee5cc95755ba7ba0fb9c36c5e5faa
sha256sum models/Hebrew-Mistral-7B.Q4_K_M.gguf
```

## 4. Data

`data/` is not in git (participant stories). Copy the input file from
whoever has it (Dan's copy lives at
`/sci/labs/arieljaffe/dan.abergel1/narrative-creativity/data/`):

```bash
# Create the data directory
mkdir -p data

# Copy the ten-stories input file into it (adjust the source path if
# you received the file another way)
cp "/sci/labs/arieljaffe/dan.abergel1/narrative-creativity/data/ten stories with clear NTs.xlsx" data/
```

## 5. Run

```bash
# Create the directory where SLURM writes the job log
mkdir -p logs

# Submit the job: 1 GPU on the salmon partition (L40S), 2 h time limit.
# The script activates the CUDA libs and runs run_texts.py on the ten
# stories with the llamacpp backend (see run_ten_stories.sbatch)
sbatch run_ten_stories.sbatch

# Watch the queue until the job goes from PD (pending) to R (running)
squeue -u $USER

# Follow the log live — one line per story with its processing time
tail -f logs/nt_ten_stories_<jobid>.out
```

Outputs land in the project root: `ten_stories_NT.xlsx` (main, one row per
sentence with `nt_next`/`nt_all`), `ten_stories_NT_debug.xlsx`,
`ten_stories_NT_alternatives.xlsx`. Runs are deterministic for a fixed
`--seed` (default 42), so a re-run reproduces the same numbers.

## Reference run (Dan, 2026-08-14, job 45865077 — COMPLETED)

Total wall time **14 min 33 s** on one L40S GPU, model loading included.
Per-story processing time (scales with sentence count, ~10–12 s/sentence):

| # | Participant | Sentences | Time |
|---|-------------|-----------|-------|
| 1 | 14485 | 4 | 37.9 s |
| 2 | 14179 | 8 | 85.5 s |
| 3 | 14363 | 8 | 89.5 s |
| 4 | 14186 | 10 | 122.2 s |
| 5 | 14632 | 7 | 65.3 s |
| 6 | 14453 | 6 | 51.8 s |
| 7 | 14146 | 7 | 74.9 s |
| 8 | 14761 | 10 | 111.7 s |
| 9 | 14309 | 8 | 65.4 s |
| 10 | 14327 | 11 | 137.9 s |

Extrapolation for the full dataset (122 participants × 2 stories): roughly
5–6 h on a single GPU, or split across jobs with `--participants`.
