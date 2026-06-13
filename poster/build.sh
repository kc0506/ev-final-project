#!/usr/bin/env bash
# Refresh figures+numbers from gic/output, then compile the A0 poster.
set -e
cd "$(dirname "$0")"
python3 refresh.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
echo "==> poster/main.pdf"
