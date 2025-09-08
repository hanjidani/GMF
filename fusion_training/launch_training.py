#!/usr/bin/env python3
"""
Launcher script for MCN Fusion Training
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Launch MCN Fusion Training')
    parser.add_argument('--action', type=str, required=True,
                       choices=['train', 'train_all', 'evaluate', 'test'],
                       help='Action to perform')
    parser.add_argument('--fusion_type', type=str, 
                       choices=['multiplicative', 'multiplicativeAddition', 'multiplicativeShifted', 'TransformerBase', 'concatenation', 'simpleAddition'],
                       help='Type of fusion to train (required for single training)')
    parser.add_argument('--input_dim', type=int, default=None,
                       help='Input feature dimension (auto-detected if not specified)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension for MLP-based fusions (default: same as input_dim)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists('scripts'):
        print("❌ Error: This script must be run from the fusion_training/ directory")
        print("Current directory:", os.getcwd())
        return
    
    if args.action == 'train':
        if not args.fusion_type:
            print("❌ Error: --fusion_type is required for single training")
            print("\nAvailable fusion types:")
            print("  - multiplicative: Simple element-wise multiplication")
            print("  - multiplicativeAddition: Combines multiplicative + additive with MLP")
            print("  - multiplicativeShifted: Product of (LayerNorm(f_k) + 1) with MLP")
            print("  - TransformerBase: Uses attention mechanism for fusion")
            print("  - concatenation: Concatenates features and processes through MLP")
            print("  - simpleAddition: Direct feature addition with hidden layer processing")
            return
        
        print(f"🚀 Launching training for {args.fusion_type} fusion...")
        cmd = f"cd scripts && python train_fusion.py --fusion_type {args.fusion_type} --epochs {args.epochs} --batch_size {args.batch_size}"
        
        # Add input dimension if specified
        if args.input_dim:
            cmd += f" --input_dim {args.input_dim}"
        
        # Add hidden dimension for MLP-based fusions
        if args.fusion_type in ['multiplicativeAddition', 'multiplicativeShifted', 'concatenation', 'simpleAddition']:
            if args.hidden_dim:
                cmd += f" --hidden_dim {args.hidden_dim}"
            else:
                print(f"ℹ️  Using hidden_dim = input_dim for {args.fusion_type} (optimal for nonlinearity)")
        
        print(f"Running: {cmd}")
        os.system(cmd)
    
    elif args.action == 'train_all':
        print("🚀 Launching training for all fusion models...")
        print("\nTraining order:")
        print("  1. multiplicative (simple, fast)")
        print("  2. multiplicativeAddition (balanced)")
        print("  3. multiplicativeShifted (shifted product)")
        print("  4. TransformerBase (attention-based)")
        print("  5. concatenation (MLP-based)")
        print("  6. simpleAddition (addition-based)")
        os.system("cd scripts && python train_all_fusions.py")
    
    elif args.action == 'evaluate':
        print("🔍 Launching evaluation of all fusion models...")
        os.system("cd scripts && python evaluate_fusions.py")
    
    elif args.action == 'test':
        print("🧪 Testing fusion models...")
        print("\nTesting:")
        print("  - Model creation and forward pass")
        print("  - Parameter counting verification")
        print("  - Variable input dimension support")
        print("  - All fusion types: multiplicative, multiplicativeAddition, TransformerBase, concatenation")
        os.system("cd scripts && python test_fusion_models.py")
    
    print("✅ Action completed!")

if __name__ == "__main__":
    main()
