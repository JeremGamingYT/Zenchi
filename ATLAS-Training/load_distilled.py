import torch
import sys
import os
from transformers import AutoTokenizer

# Assure-toi que atlas_core.py est dans le path ou dans le dossier courant
# Note: L'import de 'atlas_core' va lancer l'installation des dépendances définie au début du fichier.
try:
    from atlas_core import ATLAS, ATLASConfig, ATLASInference, DemoTokenizer, DEVICE
except ImportError:
    print("⚠️ Impossible d'importer 'atlas_core.py'. Assurez-vous qu'il est dans le même dossier ou le PYTHONPATH.")
    # On peut essayer d'ajouter le dossier courant
    sys.path.append(os.getcwd())
    try:
        from atlas_core import ATLAS, ATLASConfig, ATLASInference, DemoTokenizer, DEVICE
    except ImportError as e:
        print(f"❌ Erreur critique d'import: {e}")
        raise

def load_distilled_model(
    checkpoint_path: str,
    device: str = None,
    use_teacher_tokenizer: bool = True,
    teacher_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
):
    """
    Charge un modèle ATLAS distillé compatible Notebook.
    
    Args:
        checkpoint_path: Chemin vers le fichier .pt (ex: 'atlas_distilled_gpt_oss.pt')
        device: 'cuda' ou 'cpu' (défaut: auto détecté par atlas_core.py)
        use_teacher_tokenizer: Si True, charge un tokenizer HuggingFace. Sinon DemoTokenizer.
        teacher_name: Nom du modèle HF pour le tokenizer (doit correspondre à celui utilisé lors de la distillation)
    
    Returns:
        inference: Une instance ATLASInference prête à l'emploi
    """
    
    current_device = device if device else DEVICE
    print(f"🔄 Chargement du modèle depuis {checkpoint_path} sur {current_device}...")

    # 1. Reconstruire la configuration
    # IMPORTANT: Doit correspondre à la config utilisée dans main() de atlas_core.py lors de l'entraînement
    print("📋 Configuration du modèle...")
    config = ATLASConfig(
        # Dimensions utilisées dans main()
        d_model=1024,
        n_layers=24,
        d_state=128,
        
        # Vocabulary
        vocab_size=50257,
        max_seq_len=4096,
        
        # Autres paramètres par défaut de main()
        certainty_threshold=0.85,
        verification_passes=3
    )
    
    # 2. Créer l'instance du modèle
    print("🔧 Instanciation de l'architecture ATLAS...")
    model = ATLAS(config)
    
    # 3. Charger les poids
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=current_device)
            
            # Gestion des différentes structures de sauvegarde
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint # Supposons que c'est le state_dict direct
            
            # Chargement strict=False pour éviter les erreurs si des buffers auxiliaires manquent
            keys = model.load_state_dict(state_dict, strict=False)
            print(f"✅ Poids chargés! (Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)})")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des poids: {e}")
            return None
    else:
        print(f"⚠️ Fichier checkpoint introuvable: {checkpoint_path}")
        return None

    model.to(current_device)
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
    print("\n🚀 Modèle prêt à l'emploi!")
    return inference

# ==========================================
# Exemple d'utilisation pour Notebook
# ==========================================
if __name__ == "__main__":
    # Exemple: Chargeons le modèle si le fichier existe
    distilled_path = "atlas_distilled_gpt_oss.pt"
    
    if os.path.exists(distilled_path):
        atlas = load_distilled_model(distilled_path)
        
        if atlas:
            # Test rapide
            response = atlas.answer("Pourquoi le ciel est bleu?", mode="causal")
            print(f"\n💬 Réponse:\n{response['response']}")
    else:
        print(f"Pour tester, lancez l'entraînement dans atlas_core.py d'abord pour créer {distilled_path}")
