# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="$SCRIPT_DIR/fasta/chl1_with_exon8"
OUTPUT_DIR="$SCRIPT_DIR/pdb/chl1_with_exon8"
mkdir -p $OUTPUT_DIR

# Set environment variables for large proteins to prevent OOM (Out of Memory)
export TF_FORCE_UNIFIED_MEMORY='1'
export XLA_PYTHON_CLIENT_MEM_FRACTION='0.9' # Adjust based on your VRAM (16GB VRAM)

# Add colabfold conda environment to PATH
export PATH="$HOME/localcolabfold/localcolabfold/colabfold-conda/bin:$PATH"

# Run colabfold_batch
# --use-gpu-relax: Uses CUDA for the Amber relaxation step (much faster)
# --num-recycle: Increased to 12 for the complex domain architecture of CHL1
colabfold_batch \
    --num-recycle 4 \
    --model-type alphafold2_ptm \
    --amber \
    --use-gpu-relax \
    $INPUT_DIR \
    $OUTPUT_DIR