"""
Digital Twin System for Chronic Wound Analysis
Main entry point and command-line interface
"""

import os
# Fix OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_dashboard():
    """Launch Streamlit dashboard"""
    import subprocess
    app_path = project_root / "app" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


def run_harmonize():
    """Run data harmonization"""
    from data.harmonize import DataHarmonizer
    harmonizer = DataHarmonizer()
    harmonizer.harmonize()


def run_train_yolo():
    """Train YOLO detection model"""
    from models.detection.train_yolo import YOLOTrainer
    trainer = YOLOTrainer()
    trainer.train()


def run_train_segmentation():
    """Train segmentation model"""
    from models.segmentation.train import SegmentationTrainer
    trainer = SegmentationTrainer()
    trainer.train()


def run_train_classification():
    """Train classification model"""
    from models.classification.train import ClassificationTrainer
    trainer = ClassificationTrainer()
    trainer.train()


def run_simulate_longitudinal(n_clusters: int = 40):
    """Simulate longitudinal data from CO2Wounds-V2 dataset"""
    from data.longitudinal_simulator import LongitudinalSimulator
    
    print("=" * 60)
    print("SIMULATING LONGITUDINAL WOUND DATA")
    print("=" * 60)
    print(f"Creating {n_clusters} pseudo-patient clusters...")
    
    simulator = LongitudinalSimulator(n_clusters=n_clusters)
    trajectories = simulator.run()
    
    print(f"\nSuccess! Generated {len(trajectories)} synthetic trajectories")
    print("Use these for training the diffusion model for healing simulation.")


def run_train_diffusion(epochs: int = 10, batch_size: int = 4, dataset_type: str = "severity"):
    """Train diffusion model on synthetic trajectory data"""
    from models.simulation.train_diffusion import train_diffusion
    train_diffusion(epochs=epochs, batch_size=batch_size, dataset_type=dataset_type)


def run_analyze(image_path: str, output_path: str = None):
    """Run analysis on a single image"""
    import cv2
    from pipeline.inference import InferencePipeline
    
    pipeline = InferencePipeline()
    pipeline.load()
    
    image = cv2.imread(image_path)
    result = pipeline.analyze(image)
    
    # Print results
    print("\n" + "=" * 50)
    print("WOUND ANALYSIS RESULTS")
    print("=" * 50)
    
    if result.classification:
        print(f"\nWound Type: {result.classification.wound_type}")
        print(f"Type Confidence: {result.classification.wound_type_confidence:.1%}")
        print(f"Severity: {result.classification.severity}")
    
    if result.segmentation:
        print("\nTissue Composition:")
        for tissue, pct in result.segmentation.class_percentages.items():
            print(f"  {tissue}: {pct:.1f}%")
    
    if result.risk:
        print(f"\nRisk Score: {result.risk.risk_score:.2f}")
        print(f"Risk Level: {result.risk.risk_level}")
        print("\nRecommendations:")
        for rec in result.risk.recommendations:
            print(f"  - {rec}")
    
    print(f"\nTotal Processing Time: {result.total_time:.2f}s")
    
    # Save visualization
    if output_path:
        vis = pipeline.create_report_image(result)
        cv2.imwrite(output_path, vis)
        print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Digital Twin System for Chronic Wound Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py dashboard            # Launch Streamlit dashboard
  python run.py harmonize            # Run data harmonization
  python run.py simulate-longitudinal # Create synthetic trajectories from CO2Wounds
  python run.py train-yolo           # Train YOLO detection model
  python run.py train-seg            # Train segmentation model
  python run.py train-cls            # Train classification model
  python run.py analyze image.jpg    # Analyze a single image
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    
    # Harmonize command
    subparsers.add_parser("harmonize", help="Run data harmonization")
    
    # Training commands
    subparsers.add_parser("train-yolo", help="Train YOLO detection model")
    subparsers.add_parser("train-seg", help="Train segmentation model")
    subparsers.add_parser("train-cls", help="Train classification model")
    subparsers.add_parser("train-all", help="Train all models sequentially")
    
    # Simulate longitudinal data
    sim_parser = subparsers.add_parser("simulate-longitudinal", help="Simulate longitudinal data from CO2Wounds-V2")
    sim_parser.add_argument("--clusters", "-n", type=int, default=40, help="Number of pseudo-patient clusters")
    
    # Train diffusion model
    diff_parser = subparsers.add_parser("train-diffusion", help="Train diffusion model on trajectory data")
    diff_parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    diff_parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    diff_parser.add_argument("--dataset", "-d", type=str, default="severity", 
                            choices=["severity", "pairs"], help="Dataset type")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single wound image")
    analyze_parser.add_argument("image", type=str, help="Path to wound image")
    analyze_parser.add_argument("--output", "-o", type=str, help="Output visualization path")
    
    args = parser.parse_args()
    
    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "harmonize":
        run_harmonize()
    elif args.command == "train-yolo":
        run_train_yolo()
    elif args.command == "train-seg":
        run_train_segmentation()
    elif args.command == "train-cls":
        run_train_classification()
    elif args.command == "train-all":
        print("Training all models...")
        run_train_yolo()
        run_train_segmentation()
        run_train_classification()
        print("All models trained!")
    elif args.command == "simulate-longitudinal":
        run_simulate_longitudinal(args.clusters)
    elif args.command == "train-diffusion":
        run_train_diffusion(args.epochs, args.batch_size, args.dataset)
    elif args.command == "analyze":
        run_analyze(args.image, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
