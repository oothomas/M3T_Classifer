#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Local MONAI training environment setup (conda)
# - Installs Miniconda (if missing) and initializes conda
# - Creates a conda env with PyTorch 2.5.1 (CUDA 12.4)
# - Installs core scientific stack + your requested libs
# - Clones MONAI and installs in editable mode with all extras
# - Validates PyTorch/CUDA/MONAI and key libs
# ==============================================================================

# -----------------------
# 0. CONFIG
# -----------------------
PY_VERSION="3.10"
ENV_PREFIX="${HOME}/monai_env"         # conda env path
REPO_DIR="${HOME}/MONAI"               # MONAI repo clone target
MINICONDA_DIR="${HOME}/miniconda3"     # Miniconda install path
INSTALLER_DIR="${HOME}/.tmp_installers"
mkdir -p "${INSTALLER_DIR}"

# -----------------------
# 1. Install Miniconda if needed
# -----------------------
if ! command -v conda >/dev/null 2>&1 && [ ! -d "${MINICONDA_DIR}" ]; then
  echo "Miniconda not found. Installing to ${MINICONDA_DIR} ..."
  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "${OS}" in
    Linux)
      case "${ARCH}" in
        x86_64)  INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;;
        aarch64) INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;;
        *) echo "Unsupported Linux architecture: ${ARCH}"; exit 1 ;;
      esac
      ;;
    Darwin)
      case "${ARCH}" in
        x86_64)  INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh" ;;
        arm64)   INSTALLER="Miniconda3-latest-MacOSX-arm64.sh" ;;
        *) echo "Unsupported macOS architecture: ${ARCH}"; exit 1 ;;
      esac
      ;;
    *)
      echo "Unsupported OS: ${OS}"; exit 1
      ;;
  esac

  INSTALLER_PATH="${INSTALLER_DIR}/${INSTALLER}"
  curl -L "https://repo.anaconda.com/miniconda/${INSTALLER}" -o "${INSTALLER_PATH}"
  bash "${INSTALLER_PATH}" -b -p "${MINICONDA_DIR}"
  rm -f "${INSTALLER_PATH}"
fi

# -----------------------
# 2. Initialize & activate conda
# -----------------------
# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

# Ensure shells get conda in the future
conda init bash >/dev/null 2>&1 || true

# -----------------------
# 3. Remove any previous env and repo
# -----------------------
if [ -d "${ENV_PREFIX}" ]; then
  echo "Removing old conda environment at: ${ENV_PREFIX}"
  rm -rf "${ENV_PREFIX}"
fi

if [ -d "${REPO_DIR}" ]; then
  echo "Removing old MONAI repo at: ${REPO_DIR}"
  rm -rf "${REPO_DIR}"
fi

# -----------------------
# 4. Create & activate environment (conda, not mamba)
# -----------------------
echo "Creating conda env at: ${ENV_PREFIX}"
# Note: Using conda-forge for scientific/python packages; keep pytorch/nvidia first.
conda create --yes --prefix "${ENV_PREFIX}" \
  -c pytorch -c nvidia -c conda-forge \
  python="${PY_VERSION}" \
  pip \
  git \
  ninja \
  # --- PyTorch CUDA stack ---
  pytorch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  pytorch-cuda=12.4 \
  # --- Your requested packages ---
  pandas \
  pyyaml \
  tqdm \
  scipy \
  scikit-image \
  nibabel \
  pillow \
  einops \
  wandb \
  captum \
  simpleitk

echo "Activating environment..."
conda activate "${ENV_PREFIX}"
# -----------------------
# 5. Clone MONAI
# -----------------------
echo "Cloning MONAI into: ${REPO_DIR}"
git clone https://github.com/Project-MONAI/MONAI.git "${REPO_DIR}"
cd "${REPO_DIR}"

# -----------------------
# 6. Upgrade packaging tools
# -----------------------
echo "Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

# -----------------------
# 7. Install MONAI (editable, all extras)
# -----------------------
echo "Installing MONAI in editable mode with all extras (no BUILD_MONAI=1)..."
pip install --no-build-isolation -e ".[all]"

# -----------------------
# 8. Validate installation
# -----------------------
echo ">>> PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ">>> CUDA available?"
python -c "import torch; print(torch.cuda.is_available())"
echo ">>> CUDA device count:"
python -c "import torch; print(torch.cuda.device_count())"
echo ">>> MONAI config:"
python -c "import monai; monai.config.print_config()"

echo ">>> Validating requested packages:"
python - <<'PY'
import importlib, sys
pkgs = [
  ("pandas", "pandas.__version__"),
  ("yaml", "yaml.__version__"),
  ("tqdm", "tqdm.__version__"),
  ("scipy", "scipy.__version__"),
  ("skimage", "skimage.__version__"),
  ("nibabel", "nibabel.__version__"),
  ("PIL", "PIL.__version__"),
  ("einops", "einops.__version__"),
  ("wandb", "wandb.__version__"),
  ("captum", "captum.__version__"),
  ("SimpleITK", "SimpleITK.Version_VersionString()"),
]
for mod, attr in pkgs:
    try:
        m = importlib.import_module(mod)
        ver = eval(f"{mod}.{attr.split('.',1)[1]}") if '.' in attr else getattr(m, attr)
        print(f"{mod:12s} OK  -> {ver}")
    except Exception as e:
        print(f"{mod:12s} FAIL -> {e}", file=sys.stderr)
PY

echo
echo "✅ Environment ready."
echo "To use later: conda activate ${ENV_PREFIX}"