#!/usr/bin/env bash
set -euo pipefail

git submodule add -f git@github.com:tdurieux/EnergiBridge.git EnergiBridge
git submodule add -f git@github.com:EleutherAI/lm-evaluation-harness.git lm-evaluation-harness
git submodule update --init --recursive
