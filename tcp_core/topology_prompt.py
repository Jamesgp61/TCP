import os
import sys
import json

MAP_PATH = "/workspace/topology_workspace/topology_map.json"
ROOT_DIR = "/workspace"

def get_hamming_neighbors(node_id):
    """Calculates the 6 adjacent nodes in a 6D hypercube (Hamming distance = 1)"""
    neighbors = []
    for i in range(6):
        # Flip exactly one bit at a time using XOR bitmasking
        neighbors.append(node_id ^ (1 << i))
    return neighbors

def generate_targeted_prompt(target_file, task_instruction):
    if not os.path.exists(MAP_PATH):
        print(f"[-] Error: topology_map.json not found at {MAP_PATH}")
        return

    with open(MAP_PATH, "r") as f:
        master_map = json.load(f)

    # The topology_map.json format from Node 0 contains an 'address_map' object
    address_map = master_map.get("address_map", {})
    buckets = master_map.get("buckets", {})

    if target_file not in address_map:
        # Fallback check for relative vs absolute paths
        clean_target = os.path.relpath(target_file, ROOT_DIR) if os.path.isabs(target_file) else target_file
        if clean_target not in address_map:
            print(f"[-] Error: File '{target_file}' is not currently indexed in the topology matrix.")
            return
        target_file = clean_target

    # 1. Isolate the target node coordinate
    target_node = address_map[target_file]
    print(f"[+] Targeting File: {target_file} (Sitting on Node {target_node} / {bin(target_node)[2:].zfill(6)})")

    # 2. Extract its 6 hypercube neighborhood neighbors
    neighbor_nodes = get_hamming_neighbors(target_node)
    active_neighborhood = [target_node] + neighbor_nodes

    print(f"[+] Gathering source profiles across topological neighborhood nodes: {neighbor_nodes}")

    # 3. Pull file text from the targeted neighborhood bins exclusively
    prompt_context = []
    files_gathered = 0

    for node in active_neighborhood:
        node_str = str(node)
        # Check if files exist in this node bucket
        if node_str in buckets and buckets[node_str]:
            for rel_path in buckets[node_str]:
                full_path = os.path.join(ROOT_DIR, rel_path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as file_f:
                            content = file_f.read()
                        
                        prompt_context.append(f"--- FILE: {rel_path} (Node {node}) ---\n{content}\n")
                        files_gathered += 1
                    except Exception as e:
                        pass

    # 4. Construct the Shannon-bounded prompt output string
    prompt_payload = f"""SYSTEM: You are an autonomous software architect clamped strictly to Node {target_node} of a 6D Hypercube topology matrix. 

ENVIRONMENT ENVIRONMENT PROFILE:
You have been granted high-fidelity vision ONLY into Node {target_node} and its 6 adjacent logical neighbor nodes. Any code, architecture, or configurations outside this specific subset are completely invisible to you. Do not assume imports or schemas exist unless they are explicitly present below.

ACTIVE WORKSPACE FILES ({files_gathered} files loaded into context window):
{"".join(prompt_context)}

GOAL / TASK INSTRUCTION:
Targeting '{target_file}', execute the following command precisely:
{task_instruction}

Respond with full source code modifications wrapped in clean language markdown formatting blocks. Keep your focus entirely localized.
"""

    # Output the payload directly to a staging file for the orchestrator/API client
    staging_path = f"/workspace/topology_workspace/active_prompt.txt"
    with open(staging_path, "w", encoding="utf-8") as stage_f:
        stage_f.write(prompt_payload)

    print(f"\n[+] SUCCESS: Clean prompt payload generated successfully!")
    print(f"[+] Context footprint restricted to {files_gathered} files.")
    print(f"[+] Prompt payload staged ready at: {staging_path}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 topology_prompt.py <file_path> <instruction>")
        sys.exit(1)
        
    target = sys.argv[1]
    instruction = " ".join(sys.argv[2:])
    generate_targeted_prompt(target, instruction)
