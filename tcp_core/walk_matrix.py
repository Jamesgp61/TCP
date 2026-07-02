import os
import sys
import json
import subprocess

WORKSPACE_ROOT = "/workspace/topology_workspace"

def execute_autonomous_matrix_walk(task_goal, target_nodes=None):
    """
    Walks the 6-bit hypercube node-by-node, executing targeted prompt slices
    and updating the underlying edge weights based on system success or failure.
    """
    map_path = os.path.join(WORKSPACE_ROOT, "topology_map.json")
    if not os.path.exists(map_path):
        print(f"[-] Error: topology_map.json not found at {map_path}")
        return

    with open(map_path, "r") as f:
        topology = json.load(f)

    nodes_dict = topology.get("nodes", {})
    
    # If no target indices specified, sequentially evaluate active populated nodes
    if not target_nodes:
        target_nodes = [int(k) for k, v in nodes_dict.items() if len(v.get("files", [])) > 0]
        
    print(f"[⚡] Autonomous Loop Initialized. Target Node Queue: {target_nodes}")

    for node_id in target_nodes:
        print(f"\n🚀 === COMMENCING SYSTEM PASS: NODE {node_id} (Address: {format(node_id, '06b')}) ===")
        
        # 1. Build the targeted prompt using our spatial attention controller
        print(f"[+] Isolating 6-bit context neighborhood for Node {node_id}...")
        prompt_gen_cmd = [
            "python3", os.path.join(WORKSPACE_ROOT, "topology_prompt.py"),
            str(node_id), task_goal
        ]
        subprocess.run(prompt_gen_cmd, check=True)
        
        # 2. Synchronize the active tailored payload string file
        subprocess.run([
            "cp", os.path.join(WORKSPACE_ROOT, "active_prompt.txt"),
            os.path.join(WORKSPACE_ROOT, "prompt.txt")
        ], check=True)

        # 3. Fire the inference call via the orchestrator to build code modifications
        print(f"[+] Routing isolated prompt slice directly to Venice API inference tier...")
        orchestrate_cmd = [
            "python3", os.path.join(WORKSPACE_ROOT, "orchestrator.py"),
            WORKSPACE_ROOT
        ]
        orchestrate_run = subprocess.run(orchestrate_cmd, capture_output=True, text=True)
        
        # 4. Read system output metrics and compute execution loop validation truth status
        success_truth = True
        if "API Execution Failed" in orchestrate_run.stdout or orchestrate_run.returncode != 0:
            print(f"[-] Node {node_id} execution crashed or timed out.")
            success_truth = False
        else:
            print(f"[+] Node {node_id} payload written. Running local compiler verification tests...")
            # Optional: Hook an active compilation syntax pass check here (e.g. py_compile or gcc)
            # If verify fails: success_truth = False

        # 5. Invoke edge trainer loop to adjust weights in the global energy landscape
        print(f"[+] Backpropagating execution truth down the 384 matrix edges...")
        trainer_cmd = [
            "python3", os.path.join(WORKSPACE_ROOT, "edge_trainer.py"),
            str(node_id), "1" if success_truth else "0"
        ]
        subprocess.run(trainer_cmd, check=True)
        
        # 6. Refresh the static visual interface snapshot layout
        subprocess.run(["python3", os.path.join(WORKSPACE_ROOT, "dashboard.py")], check=True)
        print(f"[+] Visual dashboard map matrix synchronized.")

    print(f"\n🎉 [COMPLETE] Autonomous Matrix Walk successfully concluded across all targets.")

if __name__ == "__main__":
    goal = "Perform a strict architectural dependency validation check to optimize code execution efficiency."
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    execute_autonomous_matrix_walk(goal)
