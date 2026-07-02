#!/usr/bin/env bash
set -e

echo "⚡ [Topological Context Protocol] Initializing Distribution Framework Setup..."

# 1. Detect target directory context
TARGET_DIR=$(pwd)
WORKSPACE_PATH="${TARGET_DIR}/topology_workspace"

echo "[+] Staging isolated workspace footprint at: ${WORKSPACE_PATH}"
mkdir -p "${WORKSPACE_PATH}/node_000000"

# 2. Inject core engine python packages directly from the distribution snapshot
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CORE_SRC="${SCRIPT_DIR}/../tcp_core"

if [ -d "${CORE_SRC}" ]; then
    cp -r "${CORE_SRC}"/* "${WORKSPACE_PATH}/"
    echo "[+] Core engine python utility matrices synced cleanly."
else
    echo "[-] Error: Distribution source files missing. Run inside package repository root."
    exit 1
fi

# 3. Create initial task prompt configuration file
cat << 'INNER_EOF' > "${WORKSPACE_PATH}/prompt.txt
SYSTEM: You are an autonomous software architect operating inside a 6-bit Topological Context framework.
TASK: Analyze the local workspace nodes to execute feature refinement constraints safely.
INNER_EOF

# 4. Trigger the local file semantic quantization pass
echo "[+] Commencing 6-bit codebase spatial quantization pass..."
python3 "${WORKSPACE_PATH}/map_project.py"

# 5. Compile the static visual dashboard interface layout
echo "[+] Compiling interactive matrix visual tracking data dashboard..."
python3 "${WORKSPACE_PATH}/dashboard.py"

echo -e "\n🎉 [SUCCESS] TCP Framework initialized perfectly inside this repository!"
echo "    -> Workspace Index: ${WORKSPACE_PATH}/topology_map.json"
echo "    -> Interactive UI Matrix: ${WORKSPACE_PATH}/topology.html"
echo "    Ready to pass localized context slices straight to your inference engines."
