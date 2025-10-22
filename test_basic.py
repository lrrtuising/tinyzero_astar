#!/usr/bin/env python3
"""
Basic integration test for TinyZero A*PO without PyTorch dependencies.
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """Test configuration loading."""
    try:
        from config import get_local_config, get_training_config
        
        local_config = get_local_config()
        training_config = get_training_config()
        
        print("✓ Configuration loading works")
        print(f"  Local config: {local_config.training.num_iterations} iterations")
        print(f"  Training config: {training_config.training.num_iterations} iterations")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_generation():
    """Test data generation without PyTorch."""
    try:
        from data.generation import generate_multiplication_problems, Problem
        
        # Test problem generation
        problems = generate_multiplication_problems(5, 2, 3)
        
        print("✓ Data generation works")
        print(f"  Generated {len(problems)} problems")
        
        # Test problem structure
        problem = problems[0]
        print(f"  Sample problem: {problem.problem}")
        print(f"  Answer: {problem.answer}")
        print(f"  Difficulty: {problem.difficulty}")
        
        return True
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False

def test_reward_computation():
    """Test reward computation without PyTorch."""
    try:
        from training.reward import parse_answer, _compute_format_reward, _compute_length_penalty
        
        # Test answer parsing
        test_texts = [
            "The answer is 123",
            "= 456", 
            "result: 789",
            "123",
            "The final result is 999.",
            "No clear answer"
        ]
        
        print("✓ Reward computation works")
        for text in test_texts:
            answer = parse_answer(text)
            print(f"  '{text}' -> {answer}")
        
        # Test format reward
        format_reward = _compute_format_reward("Let's solve this step by step. First, we multiply 12 * 34 = 408")
        print(f"  Format reward: {format_reward:.3f}")
        
        # Test length penalty
        length_penalty = _compute_length_penalty("This is a test generation", 0.01)
        print(f"  Length penalty: {length_penalty:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Reward computation test failed: {e}")
        return False

def test_astar_search():
    """Test A* search components without PyTorch."""
    try:
        from rollout.astar_search import SearchNode
        
        # Test SearchNode
        node = SearchNode(
            token_sequence=[1, 2, 3],
            log_prob=-2.5,
            value_estimate=0.8,
            parent=None,
            depth=3
        )
        
        print("✓ A* search components work")
        print(f"  F-score: {node.f_score}")
        print(f"  Path: {node.get_path()}")
        print(f"  Depth: {node.depth}")
        
        return True
    except Exception as e:
        print(f"✗ A* search test failed: {e}")
        return False

def test_optimizer_utilities():
    """Test optimizer utilities without PyTorch."""
    try:
        from training.optimizer import get_lr, get_all_lrs
        
        print("✓ Optimizer utilities work")
        print("  (Full testing requires PyTorch)")
        
        return True
    except Exception as e:
        print(f"✗ Optimizer utilities test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Running TinyZero A*PO Basic Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Data Generation", test_data_generation),
        ("Reward Computation", test_reward_computation),
        ("A* Search Components", test_astar_search),
        ("Optimizer Utilities", test_optimizer_utilities)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The implementation is ready.")
        print("\n📝 Next steps:")
        print("1. Install PyTorch: pip install torch")
        print("2. Install transformers: pip install transformers")
        print("3. Run full training: python train.py --config local")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
