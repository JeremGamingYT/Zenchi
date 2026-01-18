import os

def create_notebook_script():
    source_file = "atlas_core_v2.py"
    target_file = "atlas_v2_notebook.py"
    
    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found.")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(target_file, 'w', encoding='utf-8') as f:
        # Add header
        f.write("# ------------------------------------------------------------------\n")
        f.write("# ATLAS V2 MODEL - NOTEBOOK COMPATIBLE VERSION\n")
        f.write("# Auto-generated from atlas_core_v2.py\n")
        f.write("# Contains: Core Model + Loader + Test Suite\n")
        f.write("# ------------------------------------------------------------------\n\n")

        # Process lines to disable main execution loop but keep classes
        for line in lines:
            # Disable main execution check
            if line.strip() == 'if __name__ == "__main__":':
                f.write('# if __name__ == "__main__":  # <--- DISABLED FOR NOTEBOOK IMPORT\n')
            elif line.strip() == 'model, inference = main()':
                f.write('#     model, inference = main()\n')
            else:
                f.write(line)

        # Add the Robust Loader Logic
        f.write("\n\n")
        f.write("# " + "="*70 + "\n")
        f.write("# NOTEBOOK LOADER & TEST UTILS\n")
        f.write("# " + "="*70 + "\n\n")
        
        loader_code = """
import torch
import os
from transformers import AutoTokenizer

def load_distilled_model(
    checkpoint_path: str,
    device: str = None,
    use_teacher_tokenizer: bool = True,
    teacher_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    certainty_threshold: float = 0.85
):
    \"\"\"
    Charge un modèle ATLAS distillé compatible Notebook.
    \"\"\"
    
    # Detect device if not provided
    if device is None:
        try:
             device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
             device = "cpu"

    print(f"🔄 Chargement du modèle depuis {checkpoint_path} sur {device}...")

    # 1. Charger le checkpoint d'abord pour obtenir la config
    print(f"🔄 Chargement du checkpoint depuis {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Erreur: Fichier {checkpoint_path} introuvable.")
        return None, None
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"❌ Erreur lecture checkpoint: {e}")
        return None, None

    # 2. Récupérer la configuration
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        print("✅ Configuration chargée depuis le checkpoint.")
        # Écraser ou adapter certains paramètres pour le test si nécessaire
        # config.batch_size = 1
    else:
        print("⚠️ Config non trouvée dans le checkpoint, utilisation de la config par défaut (RISQUE DE MISMATCH!)")
        config = ATLASConfig(
            vocab_size=32000, 
            d_model=1024,
            n_layers=24,
            d_state=128,
            max_seq_len=4096
        )

    # Force threshold for testing visibility
    config.certainty_threshold = 0.5
    
    # 3. Créer l'instance du modèle
    print(f"🔧 Instanciation modèle (Vocab={config.vocab_size}, Layers={config.n_layers})...")
    model = ATLAS(config)
    
    # 4. Charger les poids
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
            
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Poids chargés! (Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)})")
    
    model.to(device)
    model.eval()

    # 4. Charger le Tokenizer
    print("🔤 Configuration du tokenizer...")
    tokenizer = None
    if use_teacher_tokenizer:
        try:
            print(f"   Tentative de chargement du tokenizer HF: {teacher_name}")
            tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("   ✅ Tokenizer HF chargé.")
        except Exception as e:
            print(f"   ⚠️ Echec chargement tokenizer HF ({e}). Fallback sur DemoTokenizer.")
            
    if tokenizer is None:
        tokenizer = DemoTokenizer(vocab_size=config.vocab_size)

    return model, tokenizer

def run_notebook_tests(model_path="atlas_distilled_gpt_oss.pt"):
    model, tokenizer = load_distilled_model(model_path)
    
    if model is None:
        return
        
    inference = ATLASInference(model, tokenizer)
    
    test_cases = [
        "Question: If I have 3 apples and eat one, how many do I have?\\nReasoning:",
        "Write a Python function to calculate the factorial of n.\\n```python\\n",
        "User: Who is the president of France?\\nAssistant:",
        "User: What is the meaning of life?\\nAssistant: Let me think about this."
    ]

    print("\\n" + "="*60)
    print("🧪 STARTING INFERENCE TESTS")
    print("="*60)

    for i, prompt in enumerate(test_cases):
        print(f"\\n🔹 Test {i+1}:")
        print(f"Prompt: {prompt.strip()}")
        print("-" * 30)
        
        try:
            output = inference.answer(prompt, verbose=False)
            print(f"🤖 ATLAS: {output['response']}")
            print(f"📊 Confidence: {output['confidence']:.2f}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\\n✅ TESTS COMPLETED")
"""
        f.write(loader_code)
        
        f.write("\n\n# To run tests:\n# run_notebook_tests('atlas_distilled_gpt_oss.pt')\n")

    print(f"✅ Successfully created {target_file}")

if __name__ == "__main__":
    create_notebook_script()
