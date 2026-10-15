# sourced by the example run.sh files
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then MODE=${MODE:-gpu}; else MODE=${MODE:-cpu}; fi
GPUFLAG=""; [ "$MODE" = gpu ] && GPUFLAG="--gpu"
THREADS=${THREADS:-8}
julia --project="$ROOT" -e 'using Pkg; Pkg.instantiate()'
