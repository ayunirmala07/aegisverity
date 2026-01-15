"""
AegisVerity Main Entry Point
Advanced Digital Forensics Framework

Usage:
    python main.py --input <media_file> [--output <output_dir>] [--config <config_file>]
"""

import argparse
import sys
import time
import json
from pathlib import Path

from core import ForensicPipeline
from core.data_types import DetectionConfig, MediaMetadata
from layers import (
    L1ForensicLayer,
    L2VisualLayer,
    # L3AudioVisualLayer,  # Placeholder
    # L4AudioLayer,         # Placeholder
    # L5ExplainabilityLayer, # Placeholder
    # L6FusionLayer          # Placeholder
)
from utils.media_utils import extract_media_metadata


def create_default_config() -> DetectionConfig:
    """Create default detection configuration"""
    return DetectionConfig(
        confidence_threshold=0.7,
        enable_gpu=True,
        batch_size=1,
        max_frames=100,
        sample_rate=5,
        indonesian_optimized=True,
        debug_mode=False
    )


def setup_layers(config: DetectionConfig) -> list:
    """Setup and configure detection layers"""
    layers = []
    
    # Initialize available layers
    try:
        # Layer 1: Forensic Analysis
        l1 = L1ForensicLayer(config)
        layers.append(l1)
        print("✅ L1 Forensic Analysis Layer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize L1: {str(e)}")
    
    try:
        # Layer 2: Visual Analysis
        l2 = L2VisualLayer(config)
        layers.append(l2)
        print("✅ L2 Visual Analysis Layer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize L2: {str(e)}")
    
    # TODO: Initialize remaining layers as they're implemented
    # L3, L4, L5, L6 would be added here
    
    return layers


def analyze_media_file(file_path: str, config: DetectionConfig, output_dir: str = None) -> dict:
    """
    Perform complete forensic analysis on media file
    
    Args:
        file_path: Path to media file
        config: Detection configuration
        output_dir: Output directory for results
        
    Returns:
        Analysis results dictionary
    """
    print(f"🛡️  STARTING AEGIS VERITY ANALYSIS 🛡️")
    print(f"📁 Input: {file_path}")
    print("=" * 50)
    
    # Extract media metadata
    print("📊 Extracting media metadata...")
    metadata = extract_media_metadata(file_path)
    print(f"   Type: {metadata.file_type}")
    print(f"   Duration: {metadata.duration}s" if metadata.duration else "N/A")
    print(f"   Resolution: {metadata.resolution}" if metadata.resolution else "N/A")
    
    # Setup detection pipeline
    print("🔧 Configuring detection layers...")
    layers = setup_layers(config)
    
    if not layers:
        print("❌ No detection layers available!")
        return {"error": "No layers initialized"}
    
    # Create forensic pipeline
    pipeline = ForensicPipeline(layers, config)
    
    # Run analysis
    print(f"🔍 Running analysis with {len(layers)} layers...")
    start_time = time.time()
    
    try:
        result = pipeline.analyze_media(file_path, metadata, parallel_execution=True)
        
        analysis_time = time.time() - start_time
        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
        
        # Display results summary
        print("\n📋 ANALYSIS RESULTS:")
        print(f"   Final Status: {result.final_status.value}")
        print(f"   Final Confidence: {result.final_confidence:.1%}")
        print(f"   Fusion Method: {result.fusion_method}")
        print(f"   Consensus Score: {result.consensus_score:.1%}")
        
        if result.supporting_evidence.get('total_anomalies', 0) > 0:
            print(f"   Anomalies Detected: {result.supporting_evidence['total_anomalies']}")
        
        # Save results
        if output_dir:
            output_file = save_analysis_results(result, file_path, output_dir)
            print(f"\n💾 Results saved to: {output_file}")
        
        # Cleanup
        pipeline.cleanup()
        
        return result.to_dict()
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        pipeline.cleanup()
        return {"error": str(e)}


def save_analysis_results(result, input_file: str, output_dir: str) -> str:
    """Save analysis results to file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    input_path = Path(input_file)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"aegisverity_{input_path.stem}_{timestamp}.json"
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    
    return str(output_file)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AegisVerity - Advanced Digital Forensics Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to media file for analysis"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for analysis results (default: ./results)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Confidence threshold for detection (default: 0.7)"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        default=True,
        help="Enable parallel layer execution (default: enabled)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Load configuration
    config = create_default_config()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
                # Update config with file settings
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            print(f"✅ Configuration loaded from: {args.config}")
        except Exception as e:
            print(f"❌ Failed to load config: {str(e)}")
            sys.exit(1)
    
    # Override with command line arguments
    config.confidence_threshold = args.threshold
    config.debug_mode = args.debug
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    results = analyze_media_file(args.input, config, args.output)
    
    # Exit with appropriate code
    if "error" in results:
        sys.exit(1)
    else:
        print("\n🎉 AEGIS VERITY ANALYSIS COMPLETE!")
        sys.exit(0)


if __name__ == "__main__":
    main()
