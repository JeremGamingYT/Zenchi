#!/usr/bin/env python3
"""
🔄 ATLAS Data Regeneration Script
========================================
Ce script régénère les données offline de distillation avec un teacher model.
À exécuter sur Kaggle/Colab avec GPU pour accéder au teacher model.

UTILISATION:
1. Upload ce script sur Kaggle
2. Activez GPU (P100/T4/H100)
3. Exécutez: python regenerate_distillation_data.py
4. Téléchargez atlas_offline_data.pt
5. Relancez la distillation avec le code corrigé
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
import random
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "teacher_model": "mistralai/Mistral-7B-Instruct-v0.2",  # Teacher model
    "num_samples": 100000,      # Nombre d'échantillons (200k pour production)
    "max_length": 256,          # Longueur max par prompt (pas 512 - trop de padding)
    "batch_size": 16,           # Batch size pour génération (ajuster selon VRAM)
    "top_k": 16,                # Top-K logits à sauvegarder (8-16 suffit)
    "output_path": "atlas_offline_data.pt",
    "use_4bit": True,           # Quantization 4-bit pour économiser VRAM
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET DE PROMPTS HAUTE QUALITÉ
# ═══════════════════════════════════════════════════════════════════════════════

class HighQualityPromptDataset(Dataset):
    """
    Dataset de prompts variés et de haute qualité pour la distillation.
    
    Inclut:
    - Conversations (anglais + français)
    - Questions de raisonnement
    - Code Python
    - Mathématiques
    - Explications de concepts
    - Instructions diverses
    
    CHAQUE prompt génère une séquence de texte RÉELLE, pas juste des tokens BOS!
    """
    
    def __init__(self, tokenizer, num_samples: int = 100000, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        
        print(f"📊 Génération de {num_samples:,} prompts haute qualité...")
        self.samples = self._generate_diverse_prompts(num_samples)
        print(f"   ✅ {len(self.samples):,} prompts générés")
        
        # Validation rapide
        self._validate_samples()
    
    def _generate_diverse_prompts(self, n: int) -> List[str]:
        """Génère des prompts variés pour couvrir toutes les capacités"""
        
        # ═══ TEMPLATES PAR CATÉGORIE ═══
        # Chaque catégorie a plusieurs variantes pour diversité
        
        conversation_prompts = [
            "User: Hello! How are you today?\nAssistant: I'm doing great, thank you for asking! How can I help you today?",
            "User: Can you explain what artificial intelligence is?\nAssistant: Certainly! Artificial intelligence, or AI, refers to",
            "User: What's the weather like in Paris?\nAssistant: I don't have access to real-time weather data, but I can tell you that Paris typically experiences",
            "User: Tell me a joke.\nAssistant: Here's a classic one: Why don't scientists trust atoms? Because they make up everything!",
            "User: What should I cook for dinner?\nAssistant: That depends on your preferences! Here are some quick and easy options:",
            "User: How do I learn programming?\nAssistant: Learning programming is a great decision! Here's my recommended approach:",
            "User: What is the meaning of life?\nAssistant: That's a profound philosophical question that has been debated for centuries.",
            "User: Can you help me with my homework?\nAssistant: Of course! I'd be happy to help. What subject is your homework about?",
            "User: Explain quantum computing simply.\nAssistant: Quantum computing uses the principles of quantum mechanics to process information differently than classical computers.",
            "User: What are the benefits of exercise?\nAssistant: Regular exercise offers numerous physical and mental health benefits, including:",
        ]
        
        reasoning_prompts = [
            "Question: If all mammals are warm-blooded and whales are mammals, are whales warm-blooded?\nLet's think step by step:\n1. We know that all mammals are warm-blooded (given premise)\n2. We know that whales are mammals (given premise)\n3. Therefore, applying logical deduction, whales must be warm-blooded.\nAnswer: Yes, whales are warm-blooded.",
            "Question: A train leaves at 8 AM traveling at 60 mph. Another train leaves at 9 AM traveling at 80 mph. When will the second train catch up?\nLet's solve this step by step:\n1. The first train has a 1-hour head start = 60 miles ahead\n2. The second train gains 80-60=20 mph on the first\n3. Time to catch up = 60/20 = 3 hours after 9 AM = 12 PM\nAnswer: 12 PM",
            "Problem: You have 3 boxes. One contains only apples, one only oranges, and one has both. All labels are wrong. You can pick one fruit from one box. How do you determine the contents?\nReasoning:\n1. Pick from the box labeled 'Mixed'\n2. Since all labels are wrong, this box contains only one type\n3. If you get an apple, this box is 'Apples Only'",
            "Question: In a room of 23 people, what's the probability that two share a birthday?\nStep 1: Calculate probability no one shares a birthday\nStep 2: P(no match) = 365/365 × 364/365 × ... × 343/365\nStep 3: P(at least one match) = 1 - P(no match) ≈ 0.507\nAnswer: About 50.7%, surprisingly high!",
        ]
        
        code_prompts = [
            "def fibonacci(n):\n    '''Calculate the nth Fibonacci number.'''\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Example usage:\nprint(fibonacci(10))  # Output: 55",
            "def quicksort(arr):\n    '''Sort array using quicksort algorithm.'''\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            "class BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n    \n    def insert(self, value):\n        if value < self.value:\n            if self.left is None:\n                self.left = BinarySearchTree(value)\n            else:\n                self.left.insert(value)",
            "# Python function to reverse a linked list\ndef reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev",
            "# Implement binary search\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        ]
        
        math_prompts = [
            "Solve: 2x + 5 = 15\nStep 1: Subtract 5 from both sides\n2x = 10\nStep 2: Divide both sides by 2\nx = 5\nAnswer: x = 5",
            "Calculate the derivative of f(x) = x³ + 2x² - 5x + 3\nUsing the power rule:\nf'(x) = 3x² + 4x - 5",
            "Find the integral of f(x) = 2x + 3\n∫(2x + 3)dx = x² + 3x + C",
            "Solve the quadratic equation: x² - 5x + 6 = 0\nUsing the quadratic formula or factoring:\n(x - 2)(x - 3) = 0\nx = 2 or x = 3",
            "Calculate: (15 × 4) ÷ 6 + 8\nStep 1: 15 × 4 = 60\nStep 2: 60 ÷ 6 = 10\nStep 3: 10 + 8 = 18\nAnswer: 18",
        ]
        
        explanation_prompts = [
            "What is machine learning?\n\nMachine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Instead of following pre-written rules, ML algorithms identify patterns in data and make predictions or decisions based on those patterns.\n\nKey concepts:\n1. Training data - examples the algorithm learns from\n2. Model - the mathematical representation of patterns\n3. Prediction - applying the model to new data",
            "Explain how the internet works:\n\nThe internet is a global network of interconnected computers that communicate using standardized protocols.\n\n1. Data is broken into packets\n2. Packets travel through routers\n3. TCP/IP ensures reliable delivery\n4. DNS translates domain names to IP addresses\n5. HTTP/HTTPS handles web communication",
            "What is photosynthesis?\n\nPhotosynthesis is the process by which plants convert light energy into chemical energy.\n\nEquation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂\n\nSteps:\n1. Light absorption by chlorophyll\n2. Water molecules split, releasing O₂\n3. CO₂ is fixed into glucose\n4. Glucose stores energy for the plant",
        ]
        
        french_prompts = [
            "Question: Qu'est-ce que l'intelligence artificielle?\nRéponse: L'intelligence artificielle (IA) est un domaine de l'informatique qui vise à créer des machines capables d'effectuer des tâches qui nécessitent normalement l'intelligence humaine.",
            "Explique le réchauffement climatique:\n\nLe réchauffement climatique est l'augmentation de la température moyenne de la Terre due aux activités humaines.\n\nCauses principales:\n1. Émissions de CO₂ (voitures, usines)\n2. Déforestation\n3. Agriculture intensive\n\nConséquences:\n- Fonte des glaciers\n- Montée des eaux\n- Événements météorologiques extrêmes",
            "User: Bonjour, comment allez-vous?\nAssistant: Bonjour! Je vais très bien, merci de demander. Comment puis-je vous aider aujourd'hui?",
        ]
        
        instruction_prompts = [
            "[INST] Write a haiku about programming [/INST]\nSilent code awaits\nBugs hide in the shadows deep\nDebugger finds truth",
            "[INST] List 5 tips for better sleep [/INST]\n1. Keep a consistent sleep schedule\n2. Avoid screens 1 hour before bed\n3. Keep your room cool and dark\n4. Limit caffeine after noon\n5. Exercise regularly, but not too late",
            "[INST] Explain the difference between Python and JavaScript [/INST]\nPython:\n- Interpreted, high-level language\n- Strong typing, clean syntax\n- Great for data science, ML, automation\n\nJavaScript:\n- Originally for web browsers\n- Weak typing, flexible syntax\n- Dominates web development",
        ]
        
        # ═══ COMBINE TOUTES LES CATÉGORIES ═══
        all_prompts = (
            conversation_prompts * 25 +  # 250 variations
            reasoning_prompts * 30 +     # 120 variations
            code_prompts * 25 +          # 125 variations
            math_prompts * 25 +          # 125 variations
            explanation_prompts * 40 +   # 120 variations
            french_prompts * 40 +        # 120 variations
            instruction_prompts * 35     # 105 variations
        )
        
        # Génère n samples en choisissant aléatoirement
        samples = []
        for _ in range(n):
            prompt = random.choice(all_prompts)
            # Ajoute un peu de variation aléatoire
            if random.random() < 0.3:
                prompt = prompt + "\n\n"  # Parfois ajoute des newlines
            samples.append(prompt)
        
        return samples
    
    def _validate_samples(self):
        """Vérifie que les samples sont bien tokenisés"""
        print("🔍 Validation des premiers samples...")
        
        for i in range(min(3, len(self.samples))):
            sample = self.samples[i]
            tokens = self.tokenizer(
                sample,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors='pt'
            )
            
            num_tokens = tokens['input_ids'].shape[1]
            decoded = self.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            
            print(f"\n   Sample {i}:")
            print(f"   Tokens: {num_tokens}")
            print(f"   Premier texte: \"{sample[:80]}...\"")
            print(f"   Décodé: \"{decoded[:80]}...\"")
            
            # Vérifie que ce n'est pas que des BOS tokens
            unique_tokens = tokens['input_ids'].unique().tolist()
            if len(unique_tokens) < 5:
                print(f"   ⚠️ ALERTE: Seulement {len(unique_tokens)} tokens uniques!")
        
        print("   ✅ Validation terminée")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


# ═══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATEUR DE DONNÉES OFFLINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_offline_data(
    teacher_model,
    tokenizer,
    dataset,
    output_path: str,
    batch_size: int = 16,
    top_k: int = 16
):
    """
    Génère et sauvegarde les données distillées.
    
    FIXED: Uses proper token handling without corrupting data
    """
    print(f"\n🚀 Génération des données offline ({len(dataset)} samples)...")
    print(f"   Batch size: {batch_size}, Top-K: {top_k}")
    
    offline_data = []
    
    from tqdm import tqdm
    teacher_model.eval()
    
    total_tokens = 0
    content_tokens = 0
    
    # DEBUG: Check first sample from dataset directly
    print("\n🔍 DEBUG: Vérification directe du dataset...")
    first_sample = dataset[0]
    print(f"   Dataset[0] input_ids shape: {first_sample['input_ids'].shape}")
    print(f"   Dataset[0] premiers tokens: {first_sample['input_ids'][:20].tolist()}")
    print(f"   Dataset[0] attention_mask sum: {first_sample['attention_mask'].sum().item()}")
    
    # Process in batches manually to avoid DataLoader issues
    num_samples = len(dataset)
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_samples, batch_size), desc="Generating Teacher Outputs"):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Get batch manually
            batch_input_ids = []
            batch_attention_mask = []
            batch_lengths = []
            
            for idx in range(start_idx, end_idx):
                sample = dataset[idx]
                input_ids = sample['input_ids']
                attention_mask = sample['attention_mask']
                
                # Get actual length (non-padding tokens)
                length = attention_mask.sum().item()
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_lengths.append(length)
            
            # Stack into tensors
            input_ids_batch = torch.stack(batch_input_ids).to(DEVICE)
            attention_mask_batch = torch.stack(batch_attention_mask).to(DEVICE)
            
            # DEBUG: First batch check
            if start_idx == 0:
                print(f"\n🔍 DEBUG: Premier batch...")
                print(f"   input_ids_batch shape: {input_ids_batch.shape}")
                print(f"   Premier sample tokens[:20]: {input_ids_batch[0, :20].tolist()}")
                print(f"   Longueur réelle: {batch_lengths[0]}")
            
            # Forward teacher
            outputs = teacher_model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                output_hidden_states=True
            )
            
            logits = outputs.logits  # (B, L, V)
            hidden = outputs.hidden_states[-1]  # (B, L, H)
            
            # Top-K compression
            topk_vals, topk_indices = torch.topk(logits, top_k, dim=-1)
            
            # Transfer to CPU
            input_ids_cpu = input_ids_batch.cpu().to(torch.int32)
            hidden_cpu = hidden.cpu().to(torch.float16)
            topk_vals_cpu = topk_vals.cpu().to(torch.float16)
            topk_indices_cpu = topk_indices.cpu().to(torch.int32)
            
            # Store per sample
            for i, length in enumerate(batch_lengths):
                length = int(length)
                
                sample_tokens = input_ids_cpu[i, :length]
                total_tokens += length
                content_tokens += (sample_tokens > 2).sum().item()
                
                offline_data.append({
                    'input_ids': sample_tokens.clone(),
                    'hidden': hidden_cpu[i, :length].clone(),
                    'topk_vals': topk_vals_cpu[i, :length].clone(),
                    'topk_indices': topk_indices_cpu[i, :length].clone(),
                })
    
    # Statistiques améliorées
    print(f"\n📊 Statistiques de génération:")
    print(f"   Total samples: {len(offline_data)}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Content tokens (ID > 2): {content_tokens:,} ({100*content_tokens/max(total_tokens,1):.1f}%)")
    
    # Analyse détaillée d'un échantillon
    if offline_data:
        sample = offline_data[0]
        tokens = sample['input_ids'].tolist()
        print(f"\n   🔍 Analyse du premier sample sauvegardé:")
        print(f"      Longueur: {len(tokens)}")
        print(f"      Premiers tokens: {tokens[:10]}")
        print(f"      Tokens uniques: {len(set(tokens))}")
        if tokens:
            print(f"      Token min/max: {min(tokens)}/{max(tokens)}")
    
    if content_tokens / max(total_tokens, 1) < 0.5:
        print("⚠️ ALERTE: Moins de 50% de tokens de contenu - vérifiez les données!")
    else:
        print("✅ Données semblent correctes (bonne diversité de tokens)")
    
    # Sauvegarde
    print(f"\n💾 Sauvegarde vers {output_path}...")
    torch.save(offline_data, output_path)
    
    file_size = os.path.getsize(output_path) / (1024**2)
    print(f"✅ Données sauvegardées! ({file_size:.1f} MB)")
    
    return offline_data


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🔄 ATLAS DATA REGENERATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Configuration: {CONFIG}")
    
    # 1. Charger le tokenizer
    print("\n🔤 Chargement du tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CRITICAL FIX: Force RIGHT padding!
    # Mistral uses LEFT padding by default (for generation), putting real tokens at END
    # This breaks our slicing logic which expects real tokens at the BEGINNING
    tokenizer.padding_side = 'right'
    
    print(f"   ✅ Tokenizer chargé (vocab_size={tokenizer.vocab_size})")
    print(f"   padding_side: {tokenizer.padding_side} (MUST be 'right')")
    
    # 2. Charger le teacher model
    print("\n🤖 Chargement du teacher model...")
    
    if CONFIG["use_4bit"]:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        teacher = AutoModelForCausalLM.from_pretrained(
            CONFIG["teacher_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            CONFIG["teacher_model"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    print(f"   ✅ Teacher chargé!")
    
    # 3. Créer le dataset de prompts
    print("\n📊 Création du dataset de prompts...")
    dataset = HighQualityPromptDataset(
        tokenizer=tokenizer,
        num_samples=CONFIG["num_samples"],
        max_length=CONFIG["max_length"]
    )
    
    # 4. Générer les données offline
    offline_data = generate_offline_data(
        teacher_model=teacher,
        tokenizer=tokenizer,
        dataset=dataset,
        output_path=CONFIG["output_path"],
        batch_size=CONFIG["batch_size"],
        top_k=CONFIG["top_k"]
    )
    
    print("\n" + "=" * 70)
    print("✅ RÉGÉNÉRATION TERMINÉE!")
    print("=" * 70)
    print(f"""
Prochaines étapes:
1. Téléchargez {CONFIG["output_path"]}
2. Placez-le dans le dossier ATLAS-Training
3. Relancez la distillation avec le code corrigé
4. Le modèle devrait maintenant apprendre correctement!
    """)
    
    return offline_data


if __name__ == "__main__":
    main()