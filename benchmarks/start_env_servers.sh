#!/usr/bin/env bash
# Start the ALFWorld (18082) and WebShop (18081) env servers for agent_evolver.
#
# The WebShop server needs a JVM (Lucene search) via pyjnius. There is no
# system JDK; use the conda OpenJDK 11 package, whose real root is lib/jvm:
#   JAVA_HOME=<pkgs>/openjdk-11.0.30-ha668962_0/lib/jvm
#
# Usage:  bash benchmarks/start_env_servers.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALF_PY="/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-alfworld/bin/python"
WS_PY="/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/envs/ee-webshop/bin/python"
JAVA_HOME="/mnt/llmshared-ssd-hd/hanguangzeng/miniconda3/pkgs/openjdk-11.0.30-ha668962_0/lib/jvm"
WEBSHOP_ROOT="/mnt/llmshared-ssd-hd/hanguangzeng/verl-agent/agent_system/environments/env_package/webshop/webshop"
ALFWORLD_DATA="/mnt/llmshared-ssd-hd/cty/data/alfworld"

start_already() {  # $1=port
  curl -s --max-time 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

if ! start_already 18082; then
  echo "[start] ALFWorld env server :18082"
  ALFWORLD_DATA="$ALFWORLD_DATA" nohup "$ALF_PY" "$REPO/benchmarks/alfworld/env_server.py" --port 18082 \
    > /tmp/alfworld_env_server.log 2>&1 &
else
  echo "[start] ALFWorld :18082 already up"
fi

if ! start_already 18081; then
  echo "[start] WebShop env server :18081"
  JAVA_HOME="$JAVA_HOME" PATH="$JAVA_HOME/bin:$PATH" WEBSHOP_ROOT="$WEBSHOP_ROOT" \
    nohup "$WS_PY" "$REPO/benchmarks/webshop/env_server.py" --port 18081 \
    > /tmp/webshop_env_server.log 2>&1 &
else
  echo "[start] WebShop :18081 already up"
fi

echo "[start] waiting for health ..."
for i in $(seq 1 60); do
  a=$(curl -s --max-time 3 http://127.0.0.1:18082/health || true)
  w=$(curl -s --max-time 3 http://127.0.0.1:18081/health || true)
  if [ -n "$a" ] && [ -n "$w" ]; then
    echo "[start] ready: alf=$a ws=$w"
    exit 0
  fi
  sleep 3
done
echo "[start] TIMEOUT waiting for env servers; check /tmp/*_env_server.log"
exit 1
