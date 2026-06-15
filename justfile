# generative-phys task runner. `just` lists recipes; `just fix` is the everyday button.

# Pin ruff so lint/format stay reproducible across machines — bump here, not ad hoc.
ruff := "uvx ruff@0.15.16"
# Python sources to touch (outputs/ _archive/ data-pd/ are gitignored — ruff skips them).
src := "reuse_mpm teacher"
# Conda/mamba env — environment.yml is the source of truth; bump deps there, not ad hoc.
env := "physdreamer"
# Conda frontend. micromamba (via mise) is what's installed locally; it's drop-in for the
# env create/update/run subcommands. Point this at "mamba"/"conda" on hosts that have them.
mamba := "micromamba"

# List available recipes.
default:
    @just --list

# Auto-fix lint issues then format — the "tidy my code" button.
fix: 
    {{ruff}} check --fix {{src}}
    {{ruff}} format {{src}}

# Sync the env to environment.yml — edit the file to add a dep, then run this.
# No --prune: keeps hand-applied post-install tweaks (link-nvrtc) that the resolver
# can't trace. Then re-link nvrtc (see link-nvrtc) since an update may replace torch.
env-update:
    {{mamba}} env update -n {{env}} -f environment.yml
    @just link-nvrtc

# Build the env fresh from environment.yml (remove first if it exists), then link nvrtc.
env-create:
    {{mamba}} env create -y -f environment.yml
    @just link-nvrtc

# Rebuild the EV poster: refresh figures+numbers from outputs/, then compile A0 -> poster/main.pdf.
poster:
    #!/usr/bin/env bash
    set -euo pipefail
    cd poster
    python3 refresh.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    echo "==> poster/main.pdf"

# Watch poster/main.tex and auto-recompile on every save (Ctrl-C to stop). Best for tweaking layout.
poster-watch:
    cd poster && latexmk -pdf -pvc -view=none -interaction=nonstopmode main.tex

# Clear poster build artifacts (keeps main.pdf).
poster-clean:
    cd poster && latexmk -c main.tex

# torch's bundled cudnn dlopen()s a bare "libnvrtc.so", but the cu118 wheel only ships
# it hash-named (libnvrtc-*.so.11.2), so cudnn convs (and thus diffusers) crash with
# "libnvrtc.so: cannot open shared object file". Symlink the unversioned name. Idempotent;
# re-run after any env (re)create or torch reinstall — the wheel never provides the soname.
link-nvrtc:
    #!/usr/bin/env bash
    set -euo pipefail
    tlib="$({{mamba}} run -n {{env}} python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
    nvrtc="$(cd "$tlib" && ls libnvrtc-*.so.11.* | head -1)"
    ln -sf "$nvrtc" "$tlib/libnvrtc.so"
    echo "linked $tlib/libnvrtc.so -> $nvrtc"
