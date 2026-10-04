#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
plot_dir="${1:-${script_dir}/results/plots}"

montage_images() {
  local tile="$1"
  local geometry="$2"
  local output="$3"
  shift 3

  if command -v magick >/dev/null 2>&1; then
    magick montage "$@" -tile "${tile}" -geometry "${geometry}" \
      -background white -gravity center "${output}"
  elif command -v montage >/dev/null 2>&1; then
    montage "$@" -tile "${tile}" -geometry "${geometry}" \
      -background white -gravity center "${output}"
  else
    echo "ImageMagick is required (install ImageMagick 7 'magick' or ImageMagick 6 'montage')." >&2
    exit 1
  fi
}

mkdir -p "${plot_dir}"
all_panels=()
for dataset in bloodmnist dermamnist pneumoniamnist pathmnist; do
  panels=("${plot_dir}/${dataset}_"*_diagnostics.png)
  if ((${#panels[@]} == 0)); then
    echo "No diagnostic panels found for ${dataset}; skipping its contact sheet."
    continue
  fi

  montage_images "2x2" "1100x1100+20+20" \
    "${plot_dir}/${dataset}_models_2x2.png" "${panels[@]}"
  all_panels+=("${panels[@]}")
  echo "Created ${dataset}_models_2x2.png (${#panels[@]} model panels)"
done

if ((${#all_panels[@]} == 0)); then
  echo "No diagnostic panel PNGs found in ${plot_dir}. Run visualize.py first." >&2
  exit 1
fi

montage_images "4x4" "650x650+16+16" \
  "${plot_dir}/all_runs_4x4.png" "${all_panels[@]}"
echo "Created all_runs_4x4.png (${#all_panels[@]} panels)"
