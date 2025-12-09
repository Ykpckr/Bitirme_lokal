#!/usr/bin/env python3
"""
ReID Module Verification Script

Verifies installation and runs basic tests for the ReID module.
"""

import sys
from pathlib import Path

# Add reid module to path
reid_root = Path(__file__).parent
sys.path.insert(0, str(reid_root))


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'pandas': 'pandas',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn',
        'ultralytics': 'Ultralytics YOLO',
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_structure():
    """Check if all required files and directories exist."""
    print("\n" + "="*60)
    print("Checking Project Structure")
    print("="*60)
    
    required_files = [
        'README.md',
        'requirements.txt',
        'configs/example_config.yaml',
        'datasets/__init__.py',
        'datasets/soccer_reid.py',
        'models/__init__.py',
        'models/backbone_resnet50.py',
        'models/head_bnneck.py',
        'losses/__init__.py',
        'losses/triplet.py',
        'engine/__init__.py',
        'engine/train.py',
        'engine/evaluate.py',
        'engine/export.py',
        'integration/__init__.py',
        'integration/embedder_infer.py',
        'integration/cost_matrix.py',
        'scripts/__init__.py',
        'scripts/make_crops_from_yolo.py',
    ]
    
    missing = []
    
    for file_path in required_files:
        full_path = reid_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠ Missing files: {len(missing)}")
        return False
    else:
        print("\n✓ All required files present!")
        return True


def test_imports():
    """Test if modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60)
    
    tests = [
        ('datasets.soccer_reid', 'Dataset module'),
        ('models.backbone_resnet50', 'Backbone module'),
        ('models.head_bnneck', 'BNNeck head module'),
        ('losses.triplet', 'Triplet loss module'),
        ('integration.embedder_infer', 'Embedder module'),
        ('integration.cost_matrix', 'Cost matrix module'),
    ]
    
    failed = []
    
    for module_name, description in tests:
        try:
            __import__(module_name)
            print(f"✓ {description}")
        except Exception as e:
            print(f"✗ {description} - ERROR: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n⚠ Failed imports: {len(failed)}")
        return False
    else:
        print("\n✓ All modules imported successfully!")
        return True


def run_unit_tests():
    """Run unit tests for key modules."""
    print("\n" + "="*60)
    print("Running Unit Tests")
    print("="*60)
    
    print("\nNote: Unit tests may take a few seconds...")
    print("Some tests may download pretrained models.\n")
    
    tests = [
        ('integration.cost_matrix', 'Cost Matrix'),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in tests:
        try:
            print(f"\nTesting {description}...")
            module = __import__(module_name, fromlist=['test_cost_matrix'])
            
            if hasattr(module, 'test_cost_matrix'):
                module.test_cost_matrix()
                print(f"✓ {description} test passed")
                passed += 1
            else:
                print(f"⚠ {description} - no test function found")
        except Exception as e:
            print(f"✗ {description} test failed: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Unit Tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


def verify_config():
    """Verify configuration file."""
    print("\n" + "="*60)
    print("Verifying Configuration")
    print("="*60)
    
    try:
        import yaml
        
        config_path = reid_root / 'configs' / 'example_config.yaml'
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Check required sections
        required_sections = [
            'model', 'loss', 'train', 'data', 'eval',
            'export', 'device', 'paths'
        ]
        
        missing = []
        for section in required_sections:
            if section in cfg:
                print(f"✓ {section}")
            else:
                print(f"✗ {section} - MISSING")
                missing.append(section)
        
        if missing:
            print(f"\n⚠ Missing config sections: {', '.join(missing)}")
            return False
        else:
            print("\n✓ Configuration is complete!")
            return True
            
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False


def print_summary(checks):
    """Print verification summary."""
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check_name}: {status}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! ReID module is ready to use.")
        print("\nNext steps:")
        print("  1. Extract player crops: python scripts/make_crops_from_yolo.py")
        print("  2. Train model: python engine/train.py")
        print("  3. Evaluate: python engine/evaluate.py")
        print("  4. See examples: python examples.py")
        print("  5. Read README.md for detailed documentation")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("Run: pip install -r requirements.txt")
    
    return all_passed


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("ReID Module Verification")
    print("="*60)
    print(f"Reid Root: {reid_root}")
    
    checks = {
        'Dependencies': check_dependencies(),
        'Project Structure': check_structure(),
        'Module Imports': test_imports(),
        'Configuration': verify_config(),
        'Unit Tests': run_unit_tests(),
    }
    
    success = print_summary(checks)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
