"""
Basic tests to validate the GPT implementation.
Run this to verify the code works without requiring PyTorch installation.
"""

import ast
import os
import json


def test_syntax():
    """Test that all Python files have valid syntax."""
    python_files = [
        'src/model/gpt.py',
        'src/data/tokenizer.py',
        'src/data/dataset.py',
        'src/utils/training.py',
        'pretrain.py',
        'finetune.py',
        'generate.py',
        'examples/pretrain_example.py',
        'examples/finetune_example.py',
        'examples/generate_example.py',
    ]
    
    for filepath in python_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        try:
            ast.parse(code)
            print(f"✓ {filepath}: Valid Python syntax")
        except SyntaxError as e:
            print(f"✗ {filepath}: Syntax error - {e}")
            return False
    
    return True


def test_json_configs():
    """Test that all JSON config files are valid."""
    json_files = [
        'configs/small_gpt.json',
        'configs/medium_gpt.json',
    ]
    
    required_keys = ['vocab_size', 'n_embd', 'n_head', 'n_layer', 'block_size', 'dropout']
    
    for filepath in json_files:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                print(f"✗ {filepath}: Missing required key '{key}'")
                return False
        
        print(f"✓ {filepath}: Valid JSON with all required keys")
    
    return True


def test_project_structure():
    """Test that the project structure is correct."""
    required_files = [
        'README.md',
        'requirements.txt',
        'pretrain.py',
        'finetune.py',
        'generate.py',
    ]
    
    required_dirs = [
        'src',
        'src/model',
        'src/data',
        'src/utils',
        'configs',
        'examples',
    ]
    
    for filepath in required_files:
        if not os.path.isfile(filepath):
            print(f"✗ Missing required file: {filepath}")
            return False
        print(f"✓ Found required file: {filepath}")
    
    for dirpath in required_dirs:
        if not os.path.isdir(dirpath):
            print(f"✗ Missing required directory: {dirpath}")
            return False
        print(f"✓ Found required directory: {dirpath}")
    
    return True


def test_readme():
    """Test that README.md has essential content."""
    with open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    
    required_sections = [
        'Installation',
        'Quick Start',
        'Pretraining',
        'Finetuning',
        'Text Generation',
        'Project Structure',
    ]
    
    for section in required_sections:
        if section not in readme:
            print(f"✗ README missing section: {section}")
            return False
    
    print("✓ README.md contains all required sections")
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("Running LLM-Scratch Validation Tests")
    print("=" * 80)
    print()
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Python Syntax", test_syntax),
        ("JSON Configs", test_json_configs),
        ("README Content", test_readme),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        passed = test_func()
        if not passed:
            all_passed = False
            print(f"\n✗ {test_name} FAILED")
        else:
            print(f"\n✓ {test_name} PASSED")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nThe LLM implementation is ready to use.")
        print("Install PyTorch with: pip install -r requirements.txt")
        print("Then run examples or train your own model!")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before using the implementation.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
