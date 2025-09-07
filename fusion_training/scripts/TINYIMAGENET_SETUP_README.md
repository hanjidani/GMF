# TinyImageNet Setup for OOD Testing

This directory contains scripts to download and set up TinyImageNet for Out-of-Distribution (OOD) detection testing in your fusion training experiments.

## Why TinyImageNet?

TinyImageNet is an excellent OOD dataset because:
- **Different domain**: Natural images but different classes than CIFAR-100
- **Realistic**: Contains real-world images, not synthetic noise
- **Challenging**: Tests model's ability to detect domain shifts
- **Standard**: Widely used in OOD detection research

## Setup Options

### Option 1: Manual Download (Recommended)

**Step 1**: Download TinyImageNet manually
```bash
# Go to Kaggle dataset page
https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

# Click "Download" button
# Save the file as "tiny-imagenet-200.zip" in fusion_training/data/
```

**Step 2**: Run the setup script
```bash
cd fusion_training/scripts
./setup_tiny_imagenet_manual.sh
```

### Option 2: Automatic Download (Alternative)

**Note**: This requires internet access and may not work due to Kaggle's authentication requirements.

```bash
cd fusion_training/scripts
./download_tiny_imagenet.sh
```

## Expected Directory Structure

After successful setup, you should have:
```
fusion_training/
├── data/
│   ├── cifar-100-python/          # CIFAR-100 dataset
│   └── tiny-imagenet-200/         # TinyImageNet dataset
│       ├── train/                  # Training images
│       ├── val/                    # Validation images
│       └── wnids.txt              # Class names
└── scripts/
    ├── train_densenet_fusions.py  # Main training script
    └── ...                        # Other scripts
```

## Verification

The setup script will verify:
- ✅ ZIP file extraction
- ✅ Directory structure
- ✅ Expected subdirectories (train/, val/)
- ✅ File permissions

## Troubleshooting

### Issue: "ZIP file not found"
**Solution**: Download TinyImageNet manually and place in `fusion_training/data/`

### Issue: "Unexpected directory structure"
**Solution**: Check if the ZIP file contains the expected folders

### Issue: "Permission denied"
**Solution**: Make scripts executable:
```bash
chmod +x setup_tiny_imagenet_manual.sh
chmod +x download_tiny_imagenet.sh
```

## Integration with Training Script

Once TinyImageNet is set up, your `train_densenet_fusions.py` script will automatically:
1. Load TinyImageNet for OOD testing
2. Compute OOD detection metrics (AUROC, AUPR, FPR95)
3. Save results to CSV files
4. Compare OOD performance across experts, baseline, and fusion models

## Expected OOD Test Results

The script will test:
- **In-distribution**: CIFAR-100 test set
- **Out-of-distribution**: TinyImageNet validation set
- **Metrics**: AUROC, AUPR, FPR95, detection accuracy

## File Sizes

- **TinyImageNet ZIP**: ~250MB
- **Extracted dataset**: ~1.2GB
- **Storage required**: ~1.5GB total

## Next Steps

After successful setup:
1. ✅ TinyImageNet is ready for OOD testing
2. 🚀 Run your fusion training experiments
3. 📊 OOD results will be automatically logged
4. 📈 Compare robustness across all models

## Support

If you encounter issues:
1. Check the error messages from the setup script
2. Verify the ZIP file integrity
3. Ensure sufficient disk space (~1.5GB)
4. Check file permissions in the data directory
