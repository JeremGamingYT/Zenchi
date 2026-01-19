#!/usr/bin/env python3
"""
🔬 ATLAS Offline Data Analyzer
Vérifie la qualité et la structure des données de distillation.
"""

import torch
import os
import sys
from collections import Counter
import numpy as np

def analyze_offline_data(data_path: str = "atlas_offline_data.pt"):
    """Analyse approfondie des données offline de distillation."""
    
    print("=" * 70)
    print("🔬 ATLAS OFFLINE DATA DIAGNOSTIC")
    print("=" * 70)
    
    if not os.path.exists(data_path):
        print(f"❌ ERREUR: Fichier {data_path} introuvable!")
        print("   Vérifiez que le fichier est dans le répertoire courant.")
        return None
    
    print(f"\n📂 Chargement de {data_path}...")
    try:
        data = torch.load(data_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None
    
    # ════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 1: Structure des données
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("📊 1. STRUCTURE DES DONNÉES")
    print("─" * 70)
    
    print(f"   Type: {type(data)}")
    print(f"   Nombre d'échantillons: {len(data)}")
    
    if len(data) == 0:
        print("❌ CRITIQUE: Aucun échantillon dans les données!")
        return None
    
    sample = data[0]
    print(f"\n   Clés du premier échantillon: {list(sample.keys())}")
    
    expected_keys = ['input_ids', 'hidden', 'topk_vals', 'topk_indices']
    missing_keys = [k for k in expected_keys if k not in sample]
    if missing_keys:
        print(f"⚠️ ALERTE: Clés manquantes: {missing_keys}")
    
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   • {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   • {key}: type={type(value)}, value={str(value)[:100]}")
    
    # ════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 2: Qualité des input_ids
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("📊 2. QUALITÉ DES INPUT_IDS (Tokens d'entrée)")
    print("─" * 70)
    
    # Sample 10 échantillons aléatoires
    sample_indices = np.random.choice(len(data), min(10, len(data)), replace=False)
    
    all_tokens = []
    seq_lengths = []
    
    for idx in sample_indices[:3]:
        tokens = data[idx]['input_ids']
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        all_tokens.extend(tokens)
        seq_lengths.append(len(tokens))
        print(f"   Sample {idx}: {tokens[:20]}... (len={len(tokens)})")
    
    # Analyse de distribution
    token_counter = Counter(all_tokens)
    most_common = token_counter.most_common(10)
    
    print(f"\n   Longueur moyenne séquence: {np.mean(seq_lengths):.1f}")
    print(f"   Tokens les plus fréquents: {most_common}")
    
    # Check pour tokens problématiques
    if most_common[0][1] / len(all_tokens) > 0.3:
        print(f"⚠️ ALERTE: Token {most_common[0][0]} représente {most_common[0][1]/len(all_tokens):.1%} des données!")
        print("   Cela peut indiquer des données de mauvaise qualité (padding excessif, tokens répétitifs).")
    
    # ════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 3: Qualité des hidden states
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("📊 3. QUALITÉ DES HIDDEN STATES (Teacher)")
    print("─" * 70)
    
    if 'hidden' in sample:
        hidden = sample['hidden']
        print(f"   Shape: {hidden.shape}")
        print(f"   Dtype: {hidden.dtype}")
        print(f"   Min/Max: {hidden.min():.4f} / {hidden.max():.4f}")
        print(f"   Mean/Std: {hidden.mean():.4f} / {hidden.std():.4f}")
        
        # Check pour NaN/Inf
        if torch.isnan(hidden).any():
            print("❌ CRITIQUE: NaN détectés dans les hidden states!")
        if torch.isinf(hidden).any():
            print("❌ CRITIQUE: Inf détectés dans les hidden states!")
        
        # Check pour variance anormale
        if hidden.std() < 0.01:
            print("⚠️ ALERTE: Variance très faible - hidden states pourraient être constants!")
        elif hidden.std() > 100:
            print("⚠️ ALERTE: Variance très élevée - hidden states pourraient être corrompus!")
    else:
        print("⚠️ ALERTE: Pas de hidden states dans les données!")
    
    # ════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 4: Qualité des logits top-k
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("📊 4. QUALITÉ DES LOGITS TOP-K (Teacher)")
    print("─" * 70)
    
    if 'topk_vals' in sample and 'topk_indices' in sample:
        topk_vals = sample['topk_vals']
        topk_indices = sample['topk_indices']
        
        print(f"   Values shape: {topk_vals.shape}")
        print(f"   Indices shape: {topk_indices.shape}")
        print(f"   K = {topk_vals.shape[-1] if len(topk_vals.shape) > 1 else topk_vals.shape[0]}")
        
        print(f"   Values Min/Max: {topk_vals.min():.4f} / {topk_vals.max():.4f}")
        print(f"   Indices Min/Max: {topk_indices.min()} / {topk_indices.max()}")
        
        # Check pour valeurs aberrantes
        if topk_vals.max() > 50 or topk_vals.min() < -50:
            print("⚠️ ALERTE: Valeurs de logits inhabituelles - vérifiez la génération!")
        
        # Check indices dans range vocab
        if topk_indices.max() >= 32000:  # Mistral vocab size
            print(f"⚠️ ALERTE: Indices > 32000 - vocab_size mismatch possible!")
        
        # Exemple de tokens top-k
        print(f"\n   Exemple premiers logits:")
        print(f"      Position 0: indices={topk_indices[0].tolist()}, vals={topk_vals[0].tolist()}")
        if len(topk_vals) > 5:
            print(f"      Position 5: indices={topk_indices[5].tolist()}, vals={topk_vals[5].tolist()}")
    else:
        print("❌ CRITIQUE: Pas de logits top-k dans les données!")
    
    # ════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 5: Cohérence des données
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("📊 5. TEST DE COHÉRENCE")
    print("─" * 70)
    
    # Vérifie que les shapes sont cohérentes
    errors = []
    for idx in sample_indices[:5]:
        s = data[idx]
        input_len = len(s['input_ids'])
        
        if 'hidden' in s:
            hidden_len = s['hidden'].shape[0]
            if input_len != hidden_len:
                errors.append(f"Sample {idx}: input_ids len={input_len} ≠ hidden len={hidden_len}")
        
        if 'topk_vals' in s:
            topk_len = s['topk_vals'].shape[0]
            if input_len != topk_len:
                errors.append(f"Sample {idx}: input_ids len={input_len} ≠ topk len={topk_len}")
    
    if errors:
        print("❌ ERREURS DE COHÉRENCE:")
        for e in errors:
            print(f"   • {e}")
    else:
        print("✅ Toutes les shapes sont cohérentes!")
    
    # ════════════════════════════════════════════════════════════════
    # DIAGNOSTIC 6: Test de décodage
    # ════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("📊 6. TEST DE DÉCODAGE (avec tokenizer)")
    print("─" * 70)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        
        for idx in sample_indices[:3]:
            tokens = data[idx]['input_ids']
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            
            # Décode
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"\n   Sample {idx}:")
            print(f"      Tokens: {tokens[:15]}...")
            print(f"      Texte: \"{text[:200]}...\"")
            
            # Check pour texte vide ou garbage
            if len(text.strip()) < 10:
                print(f"      ⚠️ ALERTE: Texte très court ou vide!")
            if text.count(text[0]) / len(text) > 0.5 and len(text) > 10:
                print(f"      ⚠️ ALERTE: Texte potentiellement répétitif!")
                
    except Exception as e:
        print(f"⚠️ Impossible de charger le tokenizer: {e}")
        print("   Installez: pip install transformers")
    
    # ════════════════════════════════════════════════════════════════
    # RÉSUMÉ
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 70)
    
    issues = []
    
    # Compte les problèmes
    if len(data) < 1000:
        issues.append("⚠️ Trop peu d'échantillons (< 1000)")
    
    if 'hidden' in sample:
        if sample['hidden'].std() < 0.01:
            issues.append("⚠️ Hidden states ont une variance trop faible")
    
    if missing_keys:
        issues.append(f"❌ Clés manquantes: {missing_keys}")
    
    if issues:
        print("⚠️ PROBLÈMES DÉTECTÉS:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n🔧 RECOMMANDATION: Régénérez les données offline avec le teacher!")
    else:
        print("✅ Aucun problème majeur détecté dans les données.")
        print("   Le problème est probablement dans le pipeline d'entraînement.")
    
    print("\n" + "=" * 70)
    print("🔍 PROCHAINES ÉTAPES SI MODÈLE RÉPÈTE DES TOKENS:")
    print("=" * 70)
    print("""
1. Si les données sont OK → Le problème est dans distill_step():
   - La reconstruction des logits denses avec -100 fausse le KL
   - Le task_loss utilise des labels mal décalés

2. Si les données sont corrompues:
   - Régénérez atlas_offline_data.pt avec le teacher
   - Vérifiez que le teacher génère du texte cohérent

3. Vérification rapide du modèle après entraînement:
   - Les logits devraient avoir une distribution variée
   - Le token le plus probable ne devrait PAS être constant
""")
    
    return data


if __name__ == "__main__":
    # Cherche le fichier dans plusieurs emplacements
    paths_to_try = [
        "atlas_offline_data.pt",
        "/workspace/atlas_offline_data.pt",
        "./atlas_offline_data.pt"
    ]
    
    data_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            data_path = p
            break
    
    if data_path:
        analyze_offline_data(data_path)
    else:
        print("❌ Fichier atlas_offline_data.pt non trouvé!")
        print("   Spécifiez le chemin: python analyze_offline_data.py <path>")
        
        if len(sys.argv) > 1:
            analyze_offline_data(sys.argv[1])