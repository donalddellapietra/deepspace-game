#!/usr/bin/env bash
set -euo pipefail

cargo run -- --render-harness "$@"
