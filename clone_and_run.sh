#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/dawitemu1/Cancer-tumor-type-prediction-and-XAI-and-Deploy-model.git"
TARGET_DIR="${1:-Cancer-tumor-type-prediction-and-XAI-and-Deploy-model}"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "Repository already exists in '$TARGET_DIR'. Pulling latest changes..."
  git -C "$TARGET_DIR" pull
else
  echo "Cloning repository into '$TARGET_DIR'..."
  git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

printf "\nAvailable notebooks in the repository:\n"
ls -1 "LLM and mL model.ipynb" "tumor_type.ipynb" 2>/dev/null || true

printf "\nInstalling Python dependencies...\n"
python3 -m pip install --upgrade pip
python3 -m pip install streamlit pandas numpy scikit-learn joblib

printf "\nTo run the Streamlit app, execute:\n"
printf "  streamlit run App_withImage.py\n"
