import os
import sys
from openai import OpenAI

def run_node_generation(node_dir):
    prompt_path = os.path.join(node_dir, "prompt.txt")
    if not os.path.exists(prompt_path):
        print(f"[-] Error: No prompt.txt found in {node_dir}")
        return False

    with open(prompt_path, "r") as f:
        prompt_content = f.read()

    # UNIVERSAL ENDPOINT CAPTURE: Pull layout from env vars with fallback defaults
    api_url = os.getenv("AI_API_URL", "https://openai.com")
    api_key = os.getenv("AI_API_KEY")
    model_name = os.getenv("AI_MODEL", "gpt-4o")

    if not api_key:
        print("[-] Error: AI_API_KEY environment variable is missing.")
        return False

    print(f"[+] Initializing universal AI API client handshake via OpenAI SDK...")
    print(f"[+] Gateway Target: {api_url}")
    print(f"[+] Active Model Profile: {model_name}")

    try:
        client = OpenAI(api_key=api_key, base_url=api_url)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=0.1,
            stream=False
        )
        
        # Agnostic container format evaluation
        if isinstance(response, str):
            ai_output = response
        elif hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                ai_output = choice.message.content
            else:
                ai_output = choice["message"]["content"]
        else:
            ai_output = str(response)
        
        if not ai_output or ai_output.strip() == "":
            print("[-] Warning: The model response payload was empty.")
            return False

        print(f"[+] Payload Received! Syncing output string logs...")
        with open(os.path.join(node_dir, "raw_response.log"), "w") as log_f:
            log_f.write(ai_output)
            
        parse_and_save_code(ai_output, node_dir)
        return True
    except Exception as e:
        print(f"\n[-] API Execution Failed: {e}")
        return False

def parse_and_save_code(markdown_text, output_dir):
    lines = markdown_text.split("\n")
    inside_block = False
    current_file = None
    file_buffer = []

    for line in lines:
        if line.startswith("```") and not inside_block and len(line) > 3:
            inside_block = True
            continue
        elif line.startswith("```") and inside_block:
            inside_block = False
            if current_file:
                target_path = os.path.join(output_dir, current_file)
                with open(target_path, "w") as f:
                    f.write("\n".join(file_buffer))
                print(f"[+] Successfully extracted: {current_file}")
            file_buffer = []
            current_file = None
            continue
        
        if inside_block:
            file_buffer.append(line)
        else:
            if any(ext in line for ext in [".py", ".h", ".c", ".cpp", ".rs"]):
                words = line.replace("`", "").replace(":", "").replace('"', '').replace("'", "").split()
                for w in words:
                    if any(w.endswith(ext) for ext in [".py", ".h", ".c", ".cpp", ".rs"]):
                        current_file = w

if __name__ == "__main__":
    target_node = "/workspace/topology_workspace/node_000000"
    if len(sys.argv) > 1:
        target_node = sys.argv[1]
        
    run_node_generation(target_node)
