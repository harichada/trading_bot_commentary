# Model Version Control System

## Overview

The trading bot now includes a comprehensive version control system for ML models. This allows you to:
- Track model performance over time
- Automatically save best-performing models
- Rollback to previous versions if needed
- Compare different model versions
- Export metrics for analysis

## How It Works

### Automatic Versioning

When the model saves (after retraining), it automatically:
1. Creates a versioned backup in `model_versions/` directory
2. Calculates performance metrics (win rate, profit, accuracy)
3. Tags the best-performing models automatically
4. Maintains up to 10 recent versions + all "best" versions

### Performance Tracking

Each version stores:
- Total trades executed
- Win rate percentage
- Total profit/loss
- Average profit per trade
- Model accuracy
- Training timestamp
- Number of training samples

## Command-Line Management

Use the `manage_model_versions.py` tool to manage versions:

### List All Versions
```bash
python manage_model_versions.py list
# Shows last 10 versions with metrics

python manage_model_versions.py list --limit 20
# Show more versions
```

### Show Best Model
```bash
python manage_model_versions.py best
# Displays the best-performing model's metrics
```

### Compare Two Versions
```bash
python manage_model_versions.py compare v_20240108_143022 v_20240108_153045
# Shows improvements and regressions between versions
```

### Rollback to Previous Version
```bash
python manage_model_versions.py rollback v_20240108_143022
# Restores a specific version as the active model
```

### Export Metrics to CSV
```bash
python manage_model_versions.py export --output model_history.csv
# Creates CSV file for analysis in Excel/pandas
```

### Clean Up Old Versions
```bash
python manage_model_versions.py cleanup --keep 5
# Keep only 5 most recent versions + best versions
```

## Integration with Trading Bot

### Automatic Versioning

The bot automatically creates versions:
- After every 100 trades
- When retraining completes
- When performance significantly improves

### Loading Specific Versions

In your Python code:
```python
# Load the best version
model.load_model()  # Automatically loads best if available

# Load a specific version
model.load_model(version_id="v_20240108_143022")
```

### Manual Version Creation

```python
# After training or significant event
model.save_model(create_version=True)
```

## Version Metadata

Each version includes:
- **ID**: Unique timestamp-based identifier
- **Metrics**: Performance statistics
- **Tags**: Labels like "best" for top performers
- **Parent**: Previous version for tracking lineage
- **Hash**: Integrity check for model file

## Best Practices

1. **Regular Checks**: Review model versions weekly
   ```bash
   python manage_model_versions.py list
   ```

2. **Before Major Changes**: Create a manual version
   ```python
   model.save_model(create_version=True)
   ```

3. **Performance Degradation**: Rollback if needed
   ```bash
   python manage_model_versions.py rollback v_[best_version_id]
   ```

4. **Analysis**: Export metrics monthly
   ```bash
   python manage_model_versions.py export
   ```

## Directory Structure

```
trading_bot_commentary/
├── scalping_ml_model.pkl          # Current active model
├── model_versions/                # Version control directory
│   ├── versions_metadata.json     # Version registry
│   ├── v_20240108_143022/        # Version directory
│   │   └── model.pkl              # Model snapshot
│   ├── v_20240108_153045/
│   │   └── model.pkl
│   └── ...
```

## Monitoring Model Evolution

Track how your model improves over time:

1. **Win Rate Trend**: See if accuracy is improving
2. **Profit Growth**: Monitor cumulative profits
3. **Stability**: Check for performance consistency
4. **Learning Rate**: Observe how quickly it adapts

## Troubleshooting

### No Versions Found
- Models need to be trained first
- Check if `model_versions/` directory exists

### Rollback Failed
- Verify version ID is correct
- Check file permissions
- Ensure model file exists

### Best Model Not Updating
- Metrics might not be improving
- Check if auto-tagging is enabled
- Verify profit calculations

## Future Enhancements

Planned features:
- A/B testing between versions
- Automatic performance alerts
- Model drift detection
- Cloud backup integration
- Web dashboard for version management