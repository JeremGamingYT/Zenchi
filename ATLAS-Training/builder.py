import os

def create_kaggle_script():
    source_file = "atlas_core.py"
    target_file = "kaggle_atlas_full.py"
    
    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found.")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(target_file, 'w', encoding='utf-8') as f:
        # Add header
        f.write("# ------------------------------------------------------------------\n")
        f.write("# ATLAS MODEL - SINGLE FILE VERSION FOR KAGGLE\n")
        f.write("# Auto-generated from atlas_core.py\n")
        f.write("# ------------------------------------------------------------------\n\n")

        # Process lines
        for line in lines:
            # Disable auto-run of installation
            if line.strip() == "install_atlas_dependencies()":
                f.write("# install_atlas_dependencies()  # <--- DISABLED FOR KAGGLE (Run manually if needed)\n")
            
            # Disable main execution
            elif line.strip() == 'if __name__ == "__main__":':
                f.write('# if __name__ == "__main__":  # <--- DISABLED MAIN EXECUTION\n')
            elif line.strip() == 'model, inference = main()':
                f.write('#     model, inference = main()\n')
            else:
                f.write(line)

        # Add the loading logic from load_distilled.py (rewritten here for simplicity)
        f.write("\n\n")
        f.write("# ------------------------------------------------------------------\n")
        f.write("# LOADER LOGIC (Merged from load_distilled.py)\n")
        f.write("# ------------------------------------------------------------------\n")
        
        loader_code = """
import sys
import os
from transformers import AutoTokenizer

# Note: ATLAS class and config are already defined in this file above.

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

    # 1. Reconstruire la configuration
    # IMPORTANT: Doit correspondre à la config utilisée lors de l'entraînement
    print("📋 Configuration du modèle...")
    config = ATLASConfig(
        # Dimensions utilisées dans main()
        d_model=1024,
        n_layers=24,
        d_state=128,
        
        # Vocabulary
        vocab_size=50257,
        max_seq_len=4096,
        
        # Autres paramètres
        certainty_threshold=certainty_threshold,
        verification_passes=3
    )
    
    # 2. Créer l'instance du modèle
    print("🔧 Instanciation de l'architecture ATLAS...")
    model = ATLAS(config)
    
    # 3. Charger les poids
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Gestion des différentes structures de sauvegarde
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Chargement strict=False pour éviter les erreurs si des buffers auxiliaires manquent
            keys = model.load_state_dict(state_dict, strict=False)
            print(f"✅ Poids chargés! (Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)})")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des poids: {e}")
            return None
    else:
        print(f"⚠️ Fichier checkpoint introuvable: {checkpoint_path}")
        print("Assurez-vous d'avoir uploadé le fichier .pt dans votre environnement Kaggle.")
        return None

    model.to(device)
    model.eval()

    # 4. Charger le Tokenizer
    print("🔤 Configuration du tokenizer...")
    if use_teacher_tokenizer:
        try:
            print(f"   Tentative de chargement du tokenizer HF: {teacher_name}")
            tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("   ✅ Tokenizer HF chargé.")
        except Exception as e:
            print(f"   ⚠️ Echec chargement tokenizer HF ({e}). Fallback sur DemoTokenizer.")
            tokenizer = DemoTokenizer(vocab_size=config.vocab_size)
    else:
        tokenizer = DemoTokenizer(vocab_size=config.vocab_size)

    # 5. Créer l'interface d'inférence
    inference = ATLASInference(model, tokenizer)
    print("\\n🚀 Modèle prêt à l'emploi!")
    return inference

# ==========================================
# Exemple d'utilisation
# ==========================================
# atlas = load_distilled_model("atlas_distilled_gpt_oss.pt", certainty_threshold=0.5)
# if atlas:
#     # Mode 'marketing' pour voir une réponse même si le modèle est nul
#     print(atlas.answer("Test", mode="causal"))

"""
        f.write(loader_code)

    print(f"✅ Successfully created {target_file}")

if __name__ == "__main__":
    create_kaggle_script()
