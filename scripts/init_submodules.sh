#!/usr/bin/env bash
set -euo pipefail

git submodule add -f git@github.com:tdurieux/EnergiBridge.git EnergiBridge
git submodule update --init --recursive
