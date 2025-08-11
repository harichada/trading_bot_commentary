#!/usr/bin/env python3
"""
Command-line tool to manage ML model versions
Usage: python manage_model_versions.py [command] [options]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from model_version_control import ModelVersionControl
import json


def list_versions(vc: ModelVersionControl, limit: int = 10):
    """List all model versions"""
    history = vc.get_version_history(limit)
    
    if not history:
        print("No model versions found.")
        return
    
    print("\n📊 Model Version History")
    print("=" * 80)
    
    for version in history:
        timestamp = datetime.fromisoformat(version['timestamp'])
        metrics = version['metrics']
        
        print(f"\n📌 Version: {version['version']}")
        print(f"   Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if version.get('is_best'):
            print(f"   🏆 BEST MODEL")
        
        if version.get('description'):
            print(f"   Description: {version['description']}")
        
        print(f"   Metrics:")
        print(f"     • Trades: {metrics.get('total_trades', 0)}")
        print(f"     • Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"     • Total Profit: ${metrics.get('total_profit', 0):.2f}")
        print(f"     • Avg Profit/Trade: ${metrics.get('avg_profit_per_trade', 0):.2f}")
        print(f"     • Model Accuracy: {metrics.get('model_accuracy', 0):.1%}")


def compare_versions(vc: ModelVersionControl, v1: str, v2: str):
    """Compare two model versions"""
    comparison = vc.compare_versions(v1, v2)
    
    if 'error' in comparison:
        print(f"Error: {comparison['error']}")
        return
    
    print(f"\n🔄 Comparing {v1} vs {v2}")
    print("=" * 80)
    
    # Show improvements
    if comparison['improvements']:
        print("\n✅ Improvements:")
        for metric, change in comparison['improvements'].items():
            print(f"   • {metric}: +{change['absolute']:.2f} ({change['percent']:+.1f}%)")
    
    # Show regressions
    if comparison['regressions']:
        print("\n⚠️ Regressions:")
        for metric, change in comparison['regressions'].items():
            print(f"   • {metric}: {change['absolute']:.2f} ({change['percent']:.1f}%)")
    
    print("\n📊 Full Metrics:")
    print(f"\n{v1}:")
    for k, v in comparison['version1']['metrics'].items():
        print(f"   • {k}: {v:.2f}" if isinstance(v, float) else f"   • {k}: {v}")
    
    print(f"\n{v2}:")
    for k, v in comparison['version2']['metrics'].items():
        print(f"   • {k}: {v:.2f}" if isinstance(v, float) else f"   • {k}: {v}")


def rollback_model(vc: ModelVersionControl, version_id: str, model_path: str):
    """Rollback to a specific model version"""
    print(f"\n🔄 Rolling back to version {version_id}...")
    
    success = vc.rollback(version_id, model_path)
    
    if success:
        print(f"✅ Successfully rolled back to version {version_id}")
        print(f"   Model saved to: {model_path}")
    else:
        print(f"❌ Failed to rollback to version {version_id}")


def export_metrics(vc: ModelVersionControl, output_path: str):
    """Export metrics to CSV"""
    vc.export_metrics_csv(output_path)
    print(f"✅ Metrics exported to {output_path}")


def show_best_model(vc: ModelVersionControl):
    """Show the best performing model"""
    best_id = vc._get_best_version()
    
    if not best_id:
        print("No best model found.")
        return
    
    best_info = vc.versions[best_id]
    metrics = best_info['metrics']
    
    print("\n🏆 Best Performing Model")
    print("=" * 80)
    print(f"Version: {best_id}")
    print(f"Created: {best_info['timestamp']}")
    print(f"\nPerformance:")
    print(f"  • Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  • Win Rate: {metrics.get('win_rate', 0):.1f}%")
    print(f"  • Total Profit: ${metrics.get('total_profit', 0):.2f}")
    print(f"  • Avg Profit/Trade: ${metrics.get('avg_profit_per_trade', 0):.2f}")
    print(f"  • Model Accuracy: {metrics.get('model_accuracy', 0):.1%}")


def cleanup_old_versions(vc: ModelVersionControl, keep_count: int):
    """Clean up old model versions"""
    print(f"\n🗑️ Cleaning up old versions (keeping {keep_count} most recent)...")
    
    initial_count = len(vc.versions)
    vc._cleanup_old_versions(keep_count)
    final_count = len(vc.versions)
    
    removed = initial_count - final_count
    print(f"✅ Removed {removed} old versions")
    print(f"   Remaining versions: {final_count}")


def main():
    parser = argparse.ArgumentParser(description="Manage ML model versions")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List model versions')
    list_parser.add_argument('--limit', type=int, default=10, help='Number of versions to show')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('version1', help='First version ID')
    compare_parser.add_argument('version2', help='Second version ID')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to a version')
    rollback_parser.add_argument('version', help='Version ID to rollback to')
    rollback_parser.add_argument('--model-path', default='scalping_ml_model.pkl', 
                                help='Path to save the model')
    
    # Best command
    best_parser = subparsers.add_parser('best', help='Show best model')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export metrics to CSV')
    export_parser.add_argument('--output', default='model_metrics.csv', 
                              help='Output CSV file path')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old versions')
    cleanup_parser.add_argument('--keep', type=int, default=10, 
                               help='Number of versions to keep')
    
    args = parser.parse_args()
    
    # Initialize version control
    vc = ModelVersionControl()
    
    if args.command == 'list':
        list_versions(vc, args.limit)
    elif args.command == 'compare':
        compare_versions(vc, args.version1, args.version2)
    elif args.command == 'rollback':
        rollback_model(vc, args.version, args.model_path)
    elif args.command == 'best':
        show_best_model(vc)
    elif args.command == 'export':
        export_metrics(vc, args.output)
    elif args.command == 'cleanup':
        cleanup_old_versions(vc, args.keep)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()