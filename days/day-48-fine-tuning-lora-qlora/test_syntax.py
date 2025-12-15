#!/usr/bin/env python3
"""
Simple syntax and import test for Day 48 exercise.py
Tests that the code is syntactically correct without requiring PyTorch.
"""

import ast
import sys

def test_syntax(filename):
    """Test if Python file has valid syntax"""
    try:
        with open(filename, 'r') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source)
        print(f"✅ {filename}: Syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filename}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {filename}: Error reading file: {e}")
        return False

def test_imports(filename):
    """Test if imports are structured correctly"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        import_lines = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
        
        print(f"📦 {filename}: Found {len(import_lines)} import statements")
        for imp in import_lines[:5]:  # Show first 5 imports
            print(f"   {imp}")
        if len(import_lines) > 5:
            print(f"   ... and {len(import_lines) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ {filename}: Error checking imports: {e}")
        return False

def test_class_definitions(filename):
    """Test if class definitions are present"""
    try:
        with open(filename, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                functions.append(node.name)
        
        print(f"🏗️  {filename}: Found {len(classes)} classes and {len(functions)} functions")
        
        expected_classes = [
            'LoRALayer', 'LoRALinear', 'LoRAConfigManager', 
            'QLoRASetup', 'MultiTaskLoRAManager', 'MemoryOptimizer', 
            'PerformanceProfiler', 'LoRADeploymentManager', 'LoRAEvaluationFramework'
        ]
        
        missing_classes = [cls for cls in expected_classes if cls not in classes]
        if missing_classes:
            print(f"⚠️  Missing classes: {missing_classes}")
        else:
            print(f"✅ All expected classes found")
        
        return len(missing_classes) == 0
        
    except Exception as e:
        print(f"❌ {filename}: Error checking classes: {e}")
        return False

def main():
    """Run all tests"""
    filename = "exercise.py"
    
    print("Day 48: Fine-tuning Techniques - LoRA & QLoRA - Syntax Test")
    print("=" * 60)
    
    tests = [
        test_syntax,
        test_imports,
        test_class_definitions
    ]
    
    results = []
    for test in tests:
        result = test(filename)
        results.append(result)
        print()
    
    if all(results):
        print("🎉 All tests passed! The exercise.py file is ready for use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())