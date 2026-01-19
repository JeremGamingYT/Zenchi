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
            vocab_size=32000, # Mistral tokenizer size
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
    
    # NETTOYAGE DES CLÉS (Fix pour torch.compile)
    # Si le modèle a été compilé, les clés ont un préfixe "_orig_mod."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v  # Enlever "_orig_mod."
        else:
            new_state_dict[k] = v
    
    state_dict = new_state_dict
            
    keys = model.load_state_dict(state_dict, strict=False)
    print(f"✅ Poids chargés! (Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)})")
    
    if len(keys.missing_keys) > 0:
        print(f"🔍 Exemples de clés manquantes: {keys.missing_keys[:5]}")
    if len(keys.unexpected_keys) > 0:
        print(f"🔍 Exemples de clés inattendues: {keys.unexpected_keys[:5]}")
    
    # =========================================================
    # TOKENIZER LOADING (STANDARD)
    # =========================================================
    model.to(device)
    model.eval()

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

def simple_generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50, repetition_penalty=1.2):
    \"\"\"
    VRAIE génération autoregressive avec anti-répétition
    \"\"\"
    model.eval()
    
    # Tokenize le prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    
    generated = input_ids.clone()
    generated_tokens = []  # Track generated tokens for repetition penalty
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            x = model.embedding(generated)
            for layer in model.layers:
                x = layer(x)
            x = model.final_norm(x)
            logits = model.output_proj(x)
            
            # Prend le logit du dernier token
            next_token_logits = logits[0, -1, :].clone()
            
            # REPETITION PENALTY: pénalise les tokens déjà générés
            for token_id in set(generated_tokens):
                next_token_logits[token_id] /= repetition_penalty
            
            # TOP-K SAMPLING: limite aux k tokens les plus probables
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Temperature scaling
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Échantillonne le prochain token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Track for repetition penalty
            generated_tokens.append(next_token.item())
            
            # Ajoute au generated
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop si EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Décode uniquement les nouveaux tokens
    new_tokens = generated[0, input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return output_text

def run_notebook_tests(model_path="atlas_distilled_gpt_oss.pt"):
    model, tokenizer = load_distilled_model(model_path)
    
    if model is None:
        return
    
    test_cases = [
        "Question: What is 2 + 2?\\nAnswer:",
        "Hello, my name is",
        "The capital of France is",
        "def factorial(n):\\n    if n == 0:\\n        return"
    ]

    print("\\n" + "="*60)
    print("🧪 STARTING AUTOREGRESSIVE GENERATION TESTS")
    print("="*60)

    for i, prompt in enumerate(test_cases):
        print(f"\\n🔹 Test {i+1}:")
        print(f"Prompt: {prompt.strip()}")
        print("-" * 30)
        
        try:
            output = simple_generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0.7)
            print(f"🤖 ATLAS: {output}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\\n✅ TESTS COMPLETED")
"""
        f.write(loader_code)
        
        f.write("\n\n# To run tests:\n# run_notebook_tests('atlas_distilled_gpt_oss.pt')\n")

    print(f"✅ Successfully created {target_file}")

if __name__ == "__main__":
    create_notebook_script()
