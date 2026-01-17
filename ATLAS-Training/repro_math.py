from atlas_core import ATLASConfig, SymbolicReasoningEngine

def test_math():
    config = ATLASConfig()
    engine = SymbolicReasoningEngine(config)
    
    print("Testing '2+2'...")
    try:
        result = engine.solve_equation("2+2")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_math()
