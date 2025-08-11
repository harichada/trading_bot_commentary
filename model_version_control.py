#!/usr/bin/env python3
"""
Model Version Control System for Trading Bot ML Models
Tracks model versions, performance metrics, and enables rollback
"""

import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import logging

logger = logging.getLogger(__name__)


class ModelVersionControl:
    """Version control system for ML models"""
    
    def __init__(self, base_dir: str = "model_versions"):
        """Initialize version control system
        
        Args:
            base_dir: Directory to store model versions
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Version metadata file
        self.metadata_file = self.base_dir / "versions_metadata.json"
        self.metadata = self._load_metadata()
        
        # Current version tracking
        self.current_version = self.metadata.get('current_version', None)
        self.versions = self.metadata.get('versions', {})
        
    def _load_metadata(self) -> Dict:
        """Load version metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'current_version': None,
            'versions': {},
            'performance_history': []
        }
    
    def _save_metadata(self):
        """Save version metadata to file"""
        self.metadata = {
            'current_version': self.current_version,
            'versions': self.versions,
            'performance_history': self.metadata.get('performance_history', [])
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID based on timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model file for integrity check"""
        with open(model_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def save_version(self, 
                    model_path: str,
                    metrics: Dict[str, float],
                    description: str = "",
                    auto_tag: bool = True) -> str:
        """Save a new model version
        
        Args:
            model_path: Path to current model file
            metrics: Performance metrics (accuracy, f1, profit, etc.)
            description: Optional description of this version
            auto_tag: Automatically tag if performance improved
            
        Returns:
            Version ID of saved model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version ID
        version_id = self._generate_version_id()
        version_dir = self.base_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Copy model file
        version_model_path = version_dir / "model.pkl"
        shutil.copy2(model_path, version_model_path)
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(version_model_path)
        
        # Determine if this is the best version
        is_best = False
        if auto_tag and self.versions:
            # Compare with previous best
            best_version = self._get_best_version()
            if best_version:
                best_metrics = self.versions[best_version]['metrics']
                # Compare primary metric (profit or accuracy)
                if metrics.get('total_profit', 0) > best_metrics.get('total_profit', 0):
                    is_best = True
            else:
                is_best = True
        elif not self.versions:
            is_best = True  # First version is best by default
        
        # Store version metadata
        version_info = {
            'id': version_id,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'metrics': metrics,
            'model_hash': model_hash,
            'model_file': str(version_model_path),
            'is_best': is_best,
            'tags': ['best'] if is_best else [],
            'parent_version': self.current_version
        }
        
        # Add to versions
        self.versions[version_id] = version_info
        self.current_version = version_id
        
        # Update performance history
        self.metadata.setdefault('performance_history', []).append({
            'version': version_id,
            'timestamp': version_info['timestamp'],
            'metrics': metrics
        })
        
        # Save metadata
        self._save_metadata()
        
        # Log the save
        logger.info(f"Saved model version {version_id}")
        if is_best:
            logger.info(f"🏆 New best model! Profit: ${metrics.get('total_profit', 0):.2f}")
        
        # Clean old versions if needed
        self._cleanup_old_versions()
        
        return version_id
    
    def load_version(self, version_id: Optional[str] = None) -> Optional[Path]:
        """Load a specific model version
        
        Args:
            version_id: Version to load (None for current/best)
            
        Returns:
            Path to model file or None if not found
        """
        if version_id is None:
            # Try to load best version first, then current
            version_id = self._get_best_version() or self.current_version
        
        if not version_id or version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return None
        
        model_path = Path(self.versions[version_id]['model_file'])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Verify integrity
        current_hash = self._calculate_model_hash(model_path)
        stored_hash = self.versions[version_id]['model_hash']
        if current_hash != stored_hash:
            logger.warning(f"Model integrity check failed for version {version_id}")
        
        logger.info(f"Loaded model version {version_id}")
        return model_path
    
    def rollback(self, version_id: str, target_model_path: str) -> bool:
        """Rollback to a specific model version
        
        Args:
            version_id: Version to rollback to
            target_model_path: Where to copy the model
            
        Returns:
            Success status
        """
        model_path = self.load_version(version_id)
        if not model_path:
            return False
        
        try:
            # Backup current model first
            target_path = Path(target_model_path)
            if target_path.exists():
                backup_path = target_path.with_suffix('.backup.pkl')
                shutil.copy2(target_path, backup_path)
                logger.info(f"Backed up current model to {backup_path}")
            
            # Copy the rollback version
            shutil.copy2(model_path, target_path)
            
            # Update current version
            self.current_version = version_id
            self._save_metadata()
            
            logger.info(f"Successfully rolled back to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare metrics between two versions
        
        Args:
            version1: First version ID
            version2: Second version ID
            
        Returns:
            Comparison dictionary
        """
        if version1 not in self.versions or version2 not in self.versions:
            return {'error': 'Version not found'}
        
        v1_info = self.versions[version1]
        v2_info = self.versions[version2]
        
        comparison = {
            'version1': {
                'id': version1,
                'timestamp': v1_info['timestamp'],
                'metrics': v1_info['metrics']
            },
            'version2': {
                'id': version2,
                'timestamp': v2_info['timestamp'],
                'metrics': v2_info['metrics']
            },
            'improvements': {},
            'regressions': {}
        }
        
        # Compare metrics
        for metric, v2_value in v2_info['metrics'].items():
            v1_value = v1_info['metrics'].get(metric, 0)
            diff = v2_value - v1_value
            pct_change = (diff / v1_value * 100) if v1_value != 0 else 0
            
            if diff > 0:
                comparison['improvements'][metric] = {
                    'absolute': diff,
                    'percent': pct_change
                }
            elif diff < 0:
                comparison['regressions'][metric] = {
                    'absolute': diff,
                    'percent': pct_change
                }
        
        return comparison
    
    def _get_best_version(self) -> Optional[str]:
        """Get the best performing version"""
        best_versions = [v_id for v_id, v_info in self.versions.items() 
                        if v_info.get('is_best', False)]
        return best_versions[-1] if best_versions else None
    
    def _cleanup_old_versions(self, keep_count: int = 10):
        """Clean up old versions, keeping only the most recent and best
        
        Args:
            keep_count: Number of versions to keep
        """
        if len(self.versions) <= keep_count:
            return
        
        # Sort versions by timestamp
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Keep the most recent versions and any tagged as 'best'
        versions_to_keep = set()
        
        # Keep recent versions
        for v_id, _ in sorted_versions[:keep_count]:
            versions_to_keep.add(v_id)
        
        # Keep best versions
        for v_id, v_info in self.versions.items():
            if v_info.get('is_best', False) or 'best' in v_info.get('tags', []):
                versions_to_keep.add(v_id)
        
        # Remove old versions
        for v_id in list(self.versions.keys()):
            if v_id not in versions_to_keep:
                # Remove files
                version_dir = self.base_dir / v_id
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                # Remove from metadata
                del self.versions[v_id]
                logger.info(f"Cleaned up old version: {v_id}")
        
        self._save_metadata()
    
    def get_version_history(self, limit: int = 10) -> List[Dict]:
        """Get version history with metrics
        
        Args:
            limit: Number of versions to return
            
        Returns:
            List of version info dictionaries
        """
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        history = []
        for v_id, v_info in sorted_versions[:limit]:
            history.append({
                'version': v_id,
                'timestamp': v_info['timestamp'],
                'description': v_info.get('description', ''),
                'metrics': v_info['metrics'],
                'is_best': v_info.get('is_best', False),
                'tags': v_info.get('tags', [])
            })
        
        return history
    
    def export_metrics_csv(self, output_path: str = "model_metrics.csv"):
        """Export version metrics to CSV for analysis
        
        Args:
            output_path: Path for CSV file
        """
        import pandas as pd
        
        data = []
        for v_id, v_info in self.versions.items():
            row = {
                'version': v_id,
                'timestamp': v_info['timestamp'],
                'description': v_info.get('description', ''),
                **v_info['metrics']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported metrics to {output_path}")


# Integration helper for existing model classes
class VersionedModel:
    """Mixin class to add version control to existing models"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version_control = ModelVersionControl()
        self.current_version_id = None
    
    def save_with_version(self, metrics: Dict[str, float], description: str = ""):
        """Save model with version control"""
        # First save normally
        self.save_model()
        
        # Then version it
        version_id = self.version_control.save_version(
            model_path=str(self.model_path),
            metrics=metrics,
            description=description
        )
        
        self.current_version_id = version_id
        return version_id
    
    def load_best_version(self):
        """Load the best performing model version"""
        model_path = self.version_control.load_version()
        if model_path:
            # Load the model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            # Update self with loaded data
            self._update_from_loaded_data(model_data)
            return True
        return False
    
    def rollback_to_version(self, version_id: str):
        """Rollback to a specific version"""
        return self.version_control.rollback(
            version_id,
            str(self.model_path)
        )


# Example usage function
def setup_model_versioning(model_instance):
    """Add version control to an existing model instance
    
    Args:
        model_instance: Model instance to add versioning to
    
    Returns:
        ModelVersionControl instance
    """
    vc = ModelVersionControl()
    model_instance.version_control = vc
    return vc