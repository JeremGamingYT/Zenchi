#!/usr/bin/env python3
"""
🧪 TEST LOCAL du tokenizer Mistral
==================================
Ce script teste le tokenizer LOCALEMENT sans GPU.
Exécute-le sur ton PC avant de lancer quoi que ce soit sur Kaggle!

UTILISATION:
    python test_tokenizer_local.py
"""

import torch
from typing import Dict, List

def test_tokenizer():
    """Test complet du tokenizer Mistral"""
    print("=" * 70)
    print("🧪 TEST LOCAL DU TOKENIZER MISTRAL")
    print("=" * 70)
    
    # 1. Charger le tokenizer (pas besoin de GPU!)
    print("\n📦 Chargement du tokenizer Mistral...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", 
            trust_remote_code=True
        )
        print(f"   ✅ Tokenizer chargé!")
        print(f"   vocab_size: {tokenizer.vocab_size}")
        print(f"   bos_token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        print(f"   eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"   pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        print("   Installez: pip install transformers")
        return False
    
    # Fix pad_token si nécessaire
    if tokenizer.pad_token is None:
        print("\n⚠️ pad_token est None, on le set à eos_token...")
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   pad_token maintenant: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # 2. Test simple de tokenization
    print("\n" + "-" * 70)
    print("📝 TEST 1: Tokenization simple (sans padding)")
    print("-" * 70)
    
    test_text = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr"
    
    tokens_no_pad = tokenizer(
        test_text,
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    print(f"   Texte: \"{test_text[:50]}...\"")
    print(f"   input_ids shape: {tokens_no_pad['input_ids'].shape}")
    print(f"   Premiers tokens: {tokens_no_pad['input_ids'][0, :15].tolist()}")
    print(f"   Tokens uniques: {len(tokens_no_pad['input_ids'][0].unique())}")
    print(f"   attention_mask sum: {tokens_no_pad['attention_mask'].sum().item()}")
    
    # Decode pour vérifier
    decoded = tokenizer.decode(tokens_no_pad['input_ids'][0], skip_special_tokens=True)
    print(f"   Décodé: \"{decoded[:50]}...\"")
    
    # 3. Test avec padding
    print("\n" + "-" * 70)
    print("📝 TEST 2: Tokenization avec padding='max_length'")
    print("-" * 70)
    
    tokens_padded = tokenizer(
        test_text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    print(f"   input_ids shape: {tokens_padded['input_ids'].shape}")
    print(f"   Premiers tokens (20): {tokens_padded['input_ids'][0, :20].tolist()}")
    print(f"   Derniers tokens (20): {tokens_padded['input_ids'][0, -20:].tolist()}")
    print(f"   attention_mask sum: {tokens_padded['attention_mask'].sum().item()}")
    
    # Vérifier la longueur réelle
    real_length = tokens_padded['attention_mask'].sum().item()
    real_tokens = tokens_padded['input_ids'][0, :int(real_length)]
    print(f"\n   Tokens réels (longueur={int(real_length)}):")
    print(f"      Premiers: {real_tokens[:10].tolist()}")
    print(f"      Derniers: {real_tokens[-5:].tolist()}")
    print(f"      Tokens uniques: {len(real_tokens.unique())}")
    
    # 4. Test du Dataset __getitem__ simulé
    print("\n" + "-" * 70)
    print("📝 TEST 3: Simulation du Dataset __getitem__")
    print("-" * 70)
    
    class TestDataset:
        def __init__(self, tokenizer, max_length=256):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.samples = [
                "def quicksort(arr):\n    '''Sort array.'''\n    if len(arr) <= 1:\n        return arr",
                "User: Hello!\nAssistant: Hi there! How can I help?",
                "Question: What is 2+2?\nAnswer: 4",
            ]
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            text = self.samples[idx]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
    
    dataset = TestDataset(tokenizer)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        length = sample['attention_mask'].sum().item()
        tokens = sample['input_ids'][:int(length)]
        
        print(f"\n   Sample {i}:")
        print(f"      Texte: \"{dataset.samples[i][:40]}...\"")
        print(f"      Longueur réelle: {int(length)}")
        print(f"      Premiers tokens: {tokens[:10].tolist()}")
        print(f"      Tokens uniques: {len(tokens.unique())}")
        print(f"      Content tokens (>2): {(tokens > 2).sum().item()}")
        
        # Vérification critique
        if tokens.unique().numel() == 1:
            print(f"      ❌ ERREUR: Tous les tokens sont identiques!")
        elif (tokens > 2).sum().item() < 5:
            print(f"      ⚠️ ALERTE: Très peu de tokens de contenu!")
        else:
            print(f"      ✅ OK: Tokens variés")
    
    # 5. Test de batching (simulation)
    print("\n" + "-" * 70)
    print("📝 TEST 4: Simulation du batching comme dans le script")
    print("-" * 70)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_lengths = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        batch_input_ids.append(sample['input_ids'])
        batch_attention_mask.append(sample['attention_mask'])
        batch_lengths.append(sample['attention_mask'].sum().item())
    
    stacked_ids = torch.stack(batch_input_ids)
    stacked_mask = torch.stack(batch_attention_mask)
    
    print(f"   Batch shape: {stacked_ids.shape}")
    
    for i in range(len(dataset)):
        length = int(batch_lengths[i])
        tokens = stacked_ids[i, :length]
        
        print(f"\n   Après stacking, sample {i}:")
        print(f"      Longueur: {length}")
        print(f"      Premiers tokens: {tokens[:10].tolist()}")
        print(f"      Tokens uniques: {len(tokens.unique())}")
        
        if tokens.unique().numel() == 1:
            print(f"      ❌ ERREUR après stacking!")
        else:
            print(f"      ✅ OK après stacking")
    
    # Résumé
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ")
    print("=" * 70)
    
    # Vérification finale
    issues = []
    
    sample = dataset[0]
    length = int(sample['attention_mask'].sum().item())
    tokens = sample['input_ids'][:length]
    
    if tokens.unique().numel() == 1:
        issues.append("❌ Tous les tokens sont identiques dans Dataset!")
    if (tokens > 2).sum().item() < length * 0.5:
        issues.append("⚠️ Moins de 50% de tokens de contenu")
    
    if issues:
        print("⚠️ PROBLÈMES DÉTECTÉS:")
        for issue in issues:
            print(f"   • {issue}")
        print("\nLe problème est dans le tokenizer ou sa configuration!")
        return False
    else:
        print("✅ TOUS LES TESTS PASSÉS!")
        print("Le tokenizer fonctionne correctement.")
        print("Le problème doit être ailleurs (GPU, mémoire, etc.)")
        return True


if __name__ == "__main__":
    test_tokenizer()
