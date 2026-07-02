# Topological Context Protocol (TCP)

TCP is a high-utility, platform-agnostic alternative to bloated AI agent frameworks and unmanaged context protocols. Instead of flooding an LLM context window with entire repository trees, TCP applies *(Lossy Semantic Vector Quantization** and **Energy-Based Model (EBM)** mathematics to map any repository into a rigid, discrete 6-bit hypercube matrix.

By enforcing a strict **6-bit (64-state) block length**, any codebase is compressed down to its maximum informational entropy. The host-side system uses this topology to surgically clamp the AI's vision exclusively to a target node and its immediate 1-bit Hamming-distance neighbors‘maximizing channel capacity and forcing 100% focused, zero-bloat reasoning.

## 📐 Mathematical Architecture

* **The 6-Bit Ceil (64 States):** Every file pathway inside the repository is deterministically mapped to a coordinate space spanning $2^6 = 64$ discrete functional bins (`000000` through `111111`).
* **Symmetrical 384-Edge Mesh:** Nodes are bound to adjacent spaces via a 6D hypercube where state transitions occur exactly at a single bit of deviation (Hamming Distance $d_H = 1$). This yields $64 \times 6 / 2 = 192$ undirected channels or exactly **384 directional topological edges**.
* **Hebbian Energy Relaxation:** System execution results (compiler passes, linter assertions, test outcomes) serve as error-driven feedback truth. Success strengthens edge weights ($W_{ij}$), expanding the local attractor basin; failure raises energy barriers to steer subsequent prompt generations away from broken code patterns.

tcp_protocol/
├── pyproject.toml              # Modern Python packaging manifest
├── bin/
│   └── setup_tcp.sh            # Universal one-line workspace initializer
├── templates/
│   ├── c_telemetry.h           # Bare-metal C data frame structure definitions
│   └── c_telemetry.c           # High-performance binary serialization logic
└── tcp_core/
    ├── adjacency.py            # Symmetrical 6D hypercube matrix configuration
    ├── embedding_map.py        # Continuous-to-discrete quantization logic
    ├── energy_model.py         # Gradient descent energy relaxation algorithms
    ├── local_embedder.py       # Zero-dependency text vectorizer
    ├── map_project.py          # Codebase file scanner and spatial hash map builder
    ├── orchestrator.py         # Provider-agnostic OpenAI SDK completion gateway
    ├── topology_prompt.py      # Surgical neighborhood context payload generator
    ├── edge_trainer.py         # Dynamic Hebbian weight optimization engine
    └── walk_matrix.py          # Continuous multi-node autonomous learning loop
---

## 🛠️ Sandbox Execution & Usage Guidelines

Before migrating files to the public distribution path (`~/tcp_protocol`), you can actively train and control the model directly inside your current `~/Project1` sandbox environment using your containerized runtime agent.

### 1. Re-index and Quantize Workspace Files
To scan your current codebase and deterministically balance files across the 64 discrete hypercube coordinates, run the project mapper script inside your active container:
```bash
docker exec -it opencode_topology_agent python3 /workspace/topology_workspace/map_project.py
```
This processes your repository assets and outputs a structured `topology_map.json` data file.

### 2. Compile the Interactive Visual Dashboard
To bake your live mapped file configurations into a clean visual matrix interface, run the dashboard script:
```bash
docker exec -it opencode_topology_agent python3 /workspace/topology_workspace/dashboard.py
```
This outputs a completely self-contained **`topology.html`** file (~77 KB) featuring inline SVG grids and interactive vanilla JavaScript tooltips. Run `cat topology_workspace/topology.html` to copy the file to your desktop browser and visually inspect file cluster densities across your 6-bit grid.

### 3. Generate a Surgical Prompt Payload
To refactor or audit a file with extreme token efficiency, use the prompt carver to isolate the target file and gather *only* its immediate topological neighbors:
```bash
docker exec -it opencode_topology_agent python3 \
  /workspace/topology_workspace/topology_prompt.py \
  manage.py \
  "Audit this script to ensure security configurations match standard production guidelines."
```
This surgically slices your codebase context window and stages a pristine prompt payload at `/workspace/topology_workspace/active_prompt.txt`.

### 4. Fire an Agnostic Inference Pass
To execute the staged prompt against any API backend (Venice, OpenAI, DeepSeek, or an offline local Ollama instance), configure your shell variables and run the client driver:
```bash
export AI_API_URL="https://venice.ai"
export AI_API_KEY="your_api_key_here"
export AI_MODEL="zai-org-glm-5-2"

# Copy active prompt over to the orchestrator tracker path
docker exec -it opencode_topology_agent cp /workspace/topology_workspace/active_prompt.txt /workspace/topology_workspace/prompt.txt

# Run the execution pass
docker exec -it opencode_topology_agent python3 /workspace/topology_workspace/orchestrator.py /workspace/topology_workspace
```
The resulting code extraction and reasoning trace logs will output cleanly to your workspace files.

---

## 🚀 Agnostic Provider Deployment Integration

The protocol client is completely decoupled from explicit platforms. You can dynamically swap inference providers instantly by passing conventional environment profiles to the runtime shell before booting **`walk_matrix.py`**:

### Scenario A: Local Offline Code Optimization (Ollama)
```bash
export AI_API_URL="http://localhost:11434/v1"
export AI_API_KEY="ollama"
export AI_MODEL="qwen2.5-coder:32b"

python3 topology_workspace/walk_matrix.py "Optimize database connection pools."
```

### Scenario B: Deep-Reasoning Cloud Infrastructure (DeepSeek)
```bash
export AI_API_URL="https://deepseek.com"
export AI_API_KEY="sk-your-deepseek-key"
export AI_MODEL="deepseek-coder"

python3 topology_workspace/walk_matrix.py "Refactor asynchronous view routing."
```

---
License: MIT
Developed for high-fidelity attention steering with zero token bloat.
