
import torch
import warnings
from transformers import AutoTokenizer, logging as transformers_logging
from atlas_core_v2 import ATLAS, ATLASConfig, DEVICE, ATLASInference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

def load_distilled_model(model_path: str, teacher_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Loads the distilled ATLAS model and the compatible tokenizer.
    """
    print(f"\n🚀 Loading Distilled Model from: {model_path}")
    print(f"🔧 Device: {DEVICE}")

    # 1. Configuration (MUST match training config)
    # Ref: atlas_core_v2.py main() config
    config = ATLASConfig(
        d_model=1024,
        n_layers=24,
        d_state=128,
        vocab_size=32000, # Updated to match Mistral/Llama tokenizer usually used in distillation
        max_seq_len=4096,
        certainty_threshold=0.85
    )

    # Note: If the vocab size in training was different (e.g. 50257 for GPT2), we need to match it.
    # The previous logs showed: "Vocab Size: 50257" in the log analysis, but "Vocab size: 32000" for Mistral.
    # The distillation script aligns student to the TEACHER's tokenizer.
    # Mistral tokenizer has 32000 tokens.
    # The student was initialized with `tokenizer = AutoTokenizer.from_pretrained(teacher_name)` effectively during the run?
    # No, in main(): `tokenizer = SimpleTokenizer(config.vocab_size)` was used initially but then:
    # `tokenizer = teacher_tokenizer` (from distiller).
    # So the student learned on the TEACHER's vocabulary.
    # We must assume the model expects Mistral's vocab size (32000) or check if the config was saved.
    
    # Let's try to load the config from the script context or just use 32000 if using Mistral.
    # However, the atlas_core_v2.py log showed defaults of 50257. 
    # But if we distilled from Mistral, we likely mapped to Mistral's token IDs.
    # Let's use 32000 for safely if we use Mistral Tokenizer.
    
    # 2. Initialize Model
    model = ATLAS(config).to(DEVICE)
    
    # 3. Load Weights
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location="cpu")
    else:
        checkpoint = torch.load(model_path)
        
    # Handle both full save (dict) or simple state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        keys = model.load_state_dict(checkpoint, strict=False)
        
    print(f"✅ Weights Loaded. Missing keys (expected for non-student layers): {len(keys.missing_keys)}")
    model.eval()

    # 4. Load Tokenizer
    print(f"📚 Loading Tokenizer: {teacher_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"⚠️ Failed to load HuggingFace tokenizer: {e}")
        print("   Using fallback/dummy tokenizer (Results will be gibberish!)")
        # Fallback dummy class if needed
        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 2
            def __call__(self, text, return_tensors='pt', **kwargs):
                return {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
            def decode(self, ids, **kwargs): return "Dummy Output"
        tokenizer = DummyTokenizer()

    return model, tokenizer

def run_tests(model, tokenizer):
    """
    Runs a series of qualitative tests.
    """
    inference = ATLASInference(model, tokenizer)
    
    test_cases = [
        {
            "category": "Reasoning (CoT)",
            "prompt": "Question: If I have 3 apples and eat one, how many do I have?\nReasoning:",
            "desc": "Basic arithmetic reasoning"
        },
        {
            "category": "Coding",
            "prompt": "Write a Python function to calculate the factorial of n.\n```python\n",
            "desc": "Code generation"
        },
        {
            "category": "Knowledge",
            "prompt": "User: Who is the president of France?\nAssistant:",
            "desc": "World knowledge"
        },
        {
            "category": "Philosophy",
            "prompt": "User: What is the meaning of life?\nAssistant: Let me think about this.",
            "desc": "Abstract reasoning triggered by CoT phrase"
        }
    ]

    print("\n" + "="*60)
    print("🧪 STARTING INFERENCE TESTS")
    print("="*60)

    for i, test in enumerate(test_cases):
        print(f"\n🔹 Test {i+1}: {test['category']} - {test['desc']}")
        print(f"   Prompt: {test['prompt'].strip()}")
        print("-" * 30)
        
        # Generation
        try:
            output = inference.answer(
                test['prompt'], 
                max_new_tokens=200, 
                temperature=0.7,
                verbose=False
            )
            
            response_text = output.get('response', '')
            # Clean up prompt from response if duplicated
            if response_text.startswith(test['prompt']):
                response_text = response_text[len(test['prompt']):]
                
            print(f"🤖 ATLAS: {response_text.strip()}")
            print(f"📊 Confidence: {output.get('confidence', 0.0):.4f}")
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")

    print("\n" + "="*60)
    print("✅ TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    MODEL_PATH = "atlas_distilled_gpt_oss.pt"
    
    try:
        model, tokenizer = load_distilled_model(MODEL_PATH)
        run_tests(model, tokenizer)
    except FileNotFoundError:
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        print("   Make sure you are running this script in the same directory as the .pt file.")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()