#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu22-zy/occlusion"
SCRIPT="${ROOT_DIR}/obstruction-preprocess/obstruction_preprocess/isaac_lab_tabletop_grasp.py"
export TERM="${TERM:-xterm}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM="xterm"
fi

if [[ -n "${ISAACLAB_ROOT:-}" && -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  ISAACLAB_SH="${ISAACLAB_ROOT}/isaaclab.sh"
elif [[ -x "${ROOT_DIR}/IsaacLab/isaaclab.sh" ]]; then
  ISAACLAB_SH="${ROOT_DIR}/IsaacLab/isaaclab.sh"
elif [[ -x "/home/ubuntu22-zy/IsaacLab/isaaclab.sh" ]]; then
  ISAACLAB_SH="/home/ubuntu22-zy/IsaacLab/isaaclab.sh"
else
  cat >&2 <<'EOF'
Isaac Lab was not found.

Set ISAACLAB_ROOT to your IsaacLab checkout, for example:

  export ISAACLAB_ROOT=/home/ubuntu22-zy/occlusion/IsaacLab

Then rerun this script.

Official docs:
  https://isaac-sim.github.io/IsaacLab/main/index.html
  https://github.com/isaac-sim/IsaacLab
EOF
  exit 2
fi

if [[ -z "${VIRTUAL_ENV:-}" && -f "${ISAACLAB_ROOT:-$(dirname "${ISAACLAB_SH}")}/env_isaaclab/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ISAACLAB_ROOT:-$(dirname "${ISAACLAB_SH}")}/env_isaaclab/bin/activate"
elif [[ -z "${VIRTUAL_ENV:-}" && -f "$(dirname "${ISAACLAB_SH}")/env_isaaclab/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$(dirname "${ISAACLAB_SH}")/env_isaaclab/bin/activate"
fi

exec "${ISAACLAB_SH}" -p "${SCRIPT}" "$@"
