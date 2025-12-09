#!/usr/bin/env python3
"""
ReID Module Main Script

Ana çalıştırma scripti - Tüm işlemleri buradan yönetebilirsiniz.
"""

import argparse
import sys
from pathlib import Path

# Add reid module to path
reid_root = Path(__file__).parent
sys.path.insert(0, str(reid_root))


def extract_crops(args, run_cfg=None):
    """Player crop'larını videolardan çıkar."""
    from scripts.make_crops_from_yolo import main as extract_main
    
    print("\n" + "="*60)
    print("🎬 PLAYER CROP EXTRACTION")
    print("="*60)
    
    # Config'den veya argument'lardan parametreleri al
    if run_cfg and 'extraction' in run_cfg:
        ext_cfg = run_cfg['extraction']
        video = args.video if hasattr(args, 'video') and args.video else ext_cfg.get('video')
        out = args.out if hasattr(args, 'out') and args.out else ext_cfg.get('output_dir')
        tracks = args.tracks if hasattr(args, 'tracks') and args.tracks else ext_cfg.get('tracks')
        conf = args.conf if hasattr(args, 'conf') and args.conf else ext_cfg.get('conf_threshold', 0.5)
        frame_interval = args.frame_interval if hasattr(args, 'frame_interval') and args.frame_interval else ext_cfg.get('frame_interval', 10)
        max_frames = args.max_frames if hasattr(args, 'max_frames') and args.max_frames else ext_cfg.get('max_frames')
    else:
        video = args.video if hasattr(args, 'video') else None
        out = args.out if hasattr(args, 'out') else None
        tracks = args.tracks if hasattr(args, 'tracks') else None
        conf = args.conf if hasattr(args, 'conf') else 0.5
        frame_interval = args.frame_interval if hasattr(args, 'frame_interval') else 10
        max_frames = args.max_frames if hasattr(args, 'max_frames') else None
    
    # Override sys.argv for the script
    old_argv = sys.argv
    sys.argv = ['make_crops_from_yolo.py']
    
    if video:
        sys.argv.extend(['--video', video])
    if out:
        sys.argv.extend(['--out', out])
    if tracks:
        sys.argv.extend(['--tracks', tracks])
    if conf is not None:
        sys.argv.extend(['--conf', str(conf)])
    if frame_interval:
        sys.argv.extend(['--frame-interval', str(frame_interval)])
    if max_frames:
        sys.argv.extend(['--max-frames', str(max_frames)])
    
    try:
        extract_main()
    finally:
        sys.argv = old_argv
    
    print("\n✅ Crop extraction tamamlandı!")


def train_model(args):
    """ReID modelini train et."""
    from engine.train import main as train_main
    
    print("\n" + "="*60)
    print("🏋️  MODEL TRAINING")
    print("="*60)
    
    # Override sys.argv
    old_argv = sys.argv
    sys.argv = ['train.py', '--cfg', args.config]
    
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    
    try:
        train_main()
    finally:
        sys.argv = old_argv
    
    print("\n✅ Training tamamlandı!")


def evaluate_model(args):
    """ReID modelini değerlendir."""
    from engine.evaluate import main as eval_main
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    # Override sys.argv
    old_argv = sys.argv
    sys.argv = ['evaluate.py', '--cfg', args.config]
    
    if args.checkpoint:
        sys.argv.extend(['--checkpoint', args.checkpoint])
    
    try:
        eval_main()
    finally:
        sys.argv = old_argv
    
    print("\n✅ Evaluation tamamlandı!")


def export_model(args):
    """ReID modelini export et."""
    from engine.export import main as export_main
    
    print("\n" + "="*60)
    print("📦 MODEL EXPORT")
    print("="*60)
    
    # Override sys.argv
    old_argv = sys.argv
    sys.argv = ['export.py', '--cfg', args.config]
    
    if args.checkpoint:
        sys.argv.extend(['--checkpoint', args.checkpoint])
    if args.test:
        sys.argv.append('--test')
    
    try:
        export_main()
    finally:
        sys.argv = old_argv
    
    print("\n✅ Export tamamlandı!")


def test_modules(args):
    """Modülleri test et."""
    import subprocess
    
    print("\n" + "="*60)
    print("🧪 MODULE TESTING")
    print("="*60)
    
    test_files = [
        'models/backbone_resnet50.py',
        'models/head_bnneck.py',
        'losses/triplet.py',
        'integration/cost_matrix.py',
    ]
    
    for test_file in test_files:
        test_path = reid_root / test_file
        print(f"\n▶️  Testing {test_file}...")
        
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=reid_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {test_file} - FAILED")
            if result.stderr:
                print(result.stderr)
    
    print("\n✅ Testing tamamlandı!")


def show_info(args):
    """Proje bilgilerini göster."""
    import yaml
    
    print("\n" + "="*60)
    print("ℹ️  ReID MODULE INFORMATION")
    print("="*60)
    
    # Load config
    config_path = reid_root / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print("\n📁 Paths:")
        print(f"  Workspace: {cfg['paths']['workspace_root']}")
        print(f"  Data: {cfg['paths']['data_root']}")
        print(f"  Output: {cfg['paths']['output_root']}")
        print(f"  YOLO Weights: {cfg['paths']['yolo_weights']}")
        
        print("\n🧠 Model:")
        print(f"  Backbone: {cfg['model']['backbone']}")
        print(f"  Embedding Dim: {cfg['model']['emb_dim']}")
        print(f"  Pretrained: {cfg['model']['pretrained']}")
        
        print("\n🎓 Training:")
        print(f"  Epochs: {cfg['train']['epochs']}")
        print(f"  Batch Size: {cfg['train']['batch_size']}")
        print(f"  Learning Rate: {cfg['train']['lr']}")
        print(f"  Optimizer: {cfg['train']['optimizer']}")
        
        print("\n📊 Data:")
        print(f"  Root: {cfg['data']['root']}")
        print(f"  Image Size: {cfg['data']['height']}x{cfg['data']['width']}")
        
        print("\n🎯 Evaluation:")
        print(f"  Metrics: {', '.join(cfg['eval']['metrics'])}")
    else:
        print(f"⚠️  Config dosyası bulunamadı: {config_path}")
    
    print("\n" + "="*60)


def full_pipeline(args):
    """Tam pipeline: Extract → Train → Evaluate → Export."""
    print("\n" + "="*60)
    print("🚀 FULL REID PIPELINE")
    print("="*60)
    print("\nAşamalar:")
    print("  1. Player crop extraction")
    print("  2. Model training")
    print("  3. Model evaluation")
    print("  4. Model export")
    print("\n" + "="*60)
    
    # 1. Extract crops
    if args.video:
        print("\n[1/4] Crop extraction başlıyor...")
        extract_crops(args)
    else:
        print("\n[1/4] Video belirtilmediği için crop extraction atlanıyor")
    
    # 2. Train
    print("\n[2/4] Training başlıyor...")
    train_model(args)
    
    # 3. Evaluate
    print("\n[3/4] Evaluation başlıyor...")
    evaluate_model(args)
    
    # 4. Export
    print("\n[4/4] Export başlıyor...")
    export_model(args)
    
    print("\n" + "="*60)
    print("🎉 FULL PIPELINE TAMAMLANDI!")
    print("="*60)
    print("\n📦 Model şurada: outputs/reid/checkpoints/best_reid.pt")
    print("\n")


def load_run_config(config_path):
    """Run config dosyasını yükle."""
    import yaml
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"⚠️  Run config bulunamadı: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description='ReID Module - Ana Çalıştırma Scripti',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Config dosyasıyla çalıştır
  python main.py run --run-config configs/run_config.yaml
  
  # Modülleri test et
  python main.py test
  
  # Bilgileri göster
  python main.py info
  
  # Crop extraction (manuel)
  python main.py extract --video match.mp4 --out data/reid
  
  # Training (manuel)
  python main.py train --config configs/example_config.yaml
  
  # Evaluation (manuel)
  python main.py eval --config configs/example_config.yaml
  
  # Export (manuel)
  python main.py export --config configs/example_config.yaml
  
  # Tam pipeline (hepsi)
  python main.py pipeline --config configs/example_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Komutlar')
    
    # Run command (config dosyasından çalıştır)
    run_parser = subparsers.add_parser(
        'run',
        help='Config dosyasından tüm parametreleri al ve çalıştır'
    )
    run_parser.add_argument(
        '--run-config',
        default='configs/run_config.yaml',
        help='Run config dosyası'
    )
    
    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Player crop extraction'
    )
    extract_parser.add_argument('--video', required=True, help='Video dosyası')
    extract_parser.add_argument('--out', required=True, help='Output klasörü')
    extract_parser.add_argument('--tracks', help='Tracking results dosyası')
    extract_parser.add_argument('--conf', type=float, default=0.5)
    extract_parser.add_argument('--frame-interval', type=int, default=10)
    extract_parser.add_argument('--max-frames', type=int)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Model training')
    train_parser.add_argument(
        '--config',
        default='configs/example_config.yaml',
        help='Config dosyası'
    )
    train_parser.add_argument('--resume', help='Resume checkpoint')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Model evaluation')
    eval_parser.add_argument(
        '--config',
        default='configs/example_config.yaml',
        help='Config dosyası'
    )
    eval_parser.add_argument('--checkpoint', help='Model checkpoint')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Model export')
    export_parser.add_argument(
        '--config',
        default='configs/example_config.yaml',
        help='Config dosyası'
    )
    export_parser.add_argument('--checkpoint', help='Model checkpoint')
    export_parser.add_argument('--test', action='store_true', help='Test after export')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test modülleri')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Proje bilgileri')
    info_parser.add_argument(
        '--config',
        default='configs/example_config.yaml',
        help='Config dosyası'
    )
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Tam pipeline')
    pipeline_parser.add_argument('--video', help='Video dosyası (opsiyonel)')
    pipeline_parser.add_argument('--out', default='data/reid', help='Output klasörü')
    pipeline_parser.add_argument('--tracks', help='Tracking results')
    pipeline_parser.add_argument(
        '--config',
        default='configs/example_config.yaml',
        help='Config dosyası'
    )
    pipeline_parser.add_argument('--resume', help='Resume checkpoint')
    pipeline_parser.add_argument('--conf', type=float, default=0.5)
    pipeline_parser.add_argument('--frame-interval', type=int, default=10)
    pipeline_parser.add_argument('--max-frames', type=int)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load run config if run command
    run_cfg = None
    if args.command == 'run':
        run_cfg = load_run_config(args.run_config)
        if run_cfg is None:
            print("❌ Run config yüklenemedi!")
            return
        
        print("\n" + "="*60)
        print("🚀 CONFIG-BASED EXECUTION")
        print("="*60)
        print(f"\nConfig: {args.run_config}\n")
    
    # Execute command
    if args.command == 'run':
        # Config'den pipeline çalıştır
        pipeline_cfg = run_cfg.get('pipeline', {})
        
        if pipeline_cfg.get('enable_extraction', True):
            print("\n[EXTRACTION] Başlıyor...")
            # Create dummy args for extraction
            class DummyArgs:
                pass
            ext_args = DummyArgs()
            extract_crops(ext_args, run_cfg)
        
        if pipeline_cfg.get('enable_training', True):
            print("\n[TRAINING] Başlıyor...")
            class DummyArgs:
                config = run_cfg.get('training', {}).get('config', 'configs/example_config.yaml')
                resume = run_cfg.get('training', {}).get('resume_checkpoint')
            train_model(DummyArgs())
        
        if pipeline_cfg.get('enable_evaluation', True):
            print("\n[EVALUATION] Başlıyor...")
            class DummyArgs:
                config = run_cfg.get('evaluation', {}).get('config', 'configs/example_config.yaml')
                checkpoint = run_cfg.get('evaluation', {}).get('checkpoint')
            evaluate_model(DummyArgs())
        
        if pipeline_cfg.get('enable_export', True):
            print("\n[EXPORT] Başlıyor...")
            class DummyArgs:
                config = run_cfg.get('export', {}).get('config', 'configs/example_config.yaml')
                checkpoint = run_cfg.get('export', {}).get('checkpoint')
                test = run_cfg.get('export', {}).get('test_after_export', True)
            export_model(DummyArgs())
        
        print("\n" + "="*60)
        print("🎉 CONFIG-BASED EXECUTION TAMAMLANDI!")
        print("="*60)
        
    elif args.command == 'extract':
        extract_crops(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'eval':
        evaluate_model(args)
    elif args.command == 'export':
        export_model(args)
    elif args.command == 'test':
        test_modules(args)
    elif args.command == 'info':
        show_info(args)
    elif args.command == 'pipeline':
        full_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
