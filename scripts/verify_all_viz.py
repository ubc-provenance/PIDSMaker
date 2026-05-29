import os
import glob
import json
import subprocess
import re

def test_visualizations():
    print("="*60)
    print("Starting Safe Visualization Check (Per-Architecture Hashes)")
    print("="*60)
    
    architectures = ["velox", "orthrus", "magic", "kairos", "flash", "nodlink", "rcaid"]
    dataset = "CADETS_E3"
    log_dir = "scripts/test_logs_CADETS_E3_20260529_092923"
    
    results = {}
    
    for arch in architectures:
        print(f"\n{'-'*60}")
        print(f"Testing architecture: {arch}")
        print(f"{'-'*60}")
        
        log_file = os.path.join(log_dir, f"{arch}.log")
        if not os.path.exists(log_file):
            print(f"⚠️ No log file found for {arch}. Skipping.")
            results[arch] = "SKIPPED (no log)"
            continue
            
        with open(log_file, "r") as f:
            content = f.read()
            
        # Extract exact hashes from the logs
        train_match = re.search(r'/home/artifacts/training/training/([a-z0-9]+)', content)
        eval_match = re.search(r'/home/artifacts/evaluation/evaluation/([a-z0-9]+)', content)
        batch_match = re.search(r'/home/artifacts/batching/batching/([a-z0-9]+)', content)
        
        if not eval_match:
            print(f"⚠️ Could not find eval hash in log for {arch}. Skipping.")
            results[arch] = "SKIPPED (no eval hash)"
            continue
            
        train_hash = train_match.group(1) if train_match else None
        eval_hash = eval_match.group(1)
        batch_hash = batch_match.group(1) if batch_match else None
        
        # Check if this architecture's specific batching cache exists
        if batch_hash:
            cache_file = f"/home/artifacts/batching/batching/{batch_hash}/CADETS_E3/preprocessed_graphs/torch_graphs.pkl"
            if not os.path.exists(cache_file):
                print(f"⚠️ Batching cache deleted for {arch} (hash: {batch_hash[:8]}...). Skipping.")
                results[arch] = "SKIPPED (cache deleted)"
                continue
            cache_dir = os.path.dirname(cache_file)
        else:
            print(f"⚠️ No batching hash found in log for {arch}. Skipping.")
            results[arch] = "SKIPPED (no batch hash)"
            continue
        
        target_manifest = f"/home/artifacts/evaluation/evaluation/{eval_hash}/CADETS_E3/viz_manifest.json"
        
        if not os.path.exists(target_manifest):
            print(f"⚠️ Manifest not found at {target_manifest}. Skipping.")
            results[arch] = "SKIPPED (no manifest)"
            continue
            
        # Patch the manifest with the correct per-architecture hashes
        with open(target_manifest, 'r') as f:
            manifest = json.load(f)
        
        manifest["preprocessed_graphs_dir"] = cache_dir
        
        if train_hash:
            trained_dir = f"/home/artifacts/training/training/{train_hash}/CADETS_E3/trained_models"
            if os.path.isdir(trained_dir):
                manifest["trained_models_dir"] = trained_dir
            else:
                manifest["trained_models_dir"] = None
                print(f"  ⚠️ Training models missing for {arch}.")
        
        with open(target_manifest, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"  [+] Train: {train_hash[:8] if train_hash else 'N/A'}... | Eval: {eval_hash[:8]}... | Batch: {batch_hash[:8]}...")
        
        # Run visualization with a tiny subset
        cmd = [
            "python", "-u", "scripts/embedding_viz.py",
            arch, dataset,
            "--embeddings", "encoder",
            "--method", "umap",
            "--max_benign", "10",
            "--max_attack", "10"
        ]
        
        process = subprocess.run(cmd)
        
        if process.returncode != 0:
            print(f"❌ {arch} FAILED!")
            results[arch] = "FAILED"
        else:
            print(f"✅ {arch} SUCCESS!")
            results[arch] = "SUCCESS"
    
    print(f"\n\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for arch, status in results.items():
        icon = "✅" if status == "SUCCESS" else ("⚠️" if "SKIPPED" in status else "❌")
        print(f"  {icon} {arch}: {status}")

if __name__ == "__main__":
    test_visualizations()
