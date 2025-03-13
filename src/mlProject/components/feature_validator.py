"""
Feature Validator Component
Validates features against saved metadata and detects feature drift.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from mlProject import logger


class FeatureValidator:
    """
    Validates features for model prediction and detects feature drift.
    Compares against saved feature statistics from training.
    """
    
    def __init__(self, metadata_path: Optional[str] = None):
        """
        Initialize feature validator
        
        Args:
            metadata_path: Path to feature metadata JSON file
        """
        self.metadata_path = metadata_path or 'artifacts/feature_metadata.json'
        self.metadata = self._load_metadata()
        logger.info(f"Initialized FeatureValidator with metadata from {self.metadata_path}")
    
    def _load_metadata(self) -> Optional[Dict]:
        """Load feature metadata from file"""
        metadata_file = Path(self.metadata_path)
        
        if not metadata_file.exists():
            logger.warning(f"Feature metadata file not found: {self.metadata_path}")
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded feature metadata: {len(metadata.get('feature_names', []))} features")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load feature metadata: {e}")
            return None
    
    def validate_feature_order(self, feature_names: list) -> Tuple[bool, str]:
        """
        Validate feature names and order against training metadata
        
        Args:
            feature_names: List of feature names to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.metadata:
            return True, "No metadata available for validation"
        
        expected_names = self.metadata.get('feature_names', [])
        
        if len(feature_names) != len(expected_names):
            return False, f"Feature count mismatch: expected {len(expected_names)}, got {len(feature_names)}"
        
        mismatches = []
        for i, (expected, actual) in enumerate(zip(expected_names, feature_names)):
            if expected != actual:
                mismatches.append(f"Index {i}: expected '{expected}', got '{actual}'")
        
        if mismatches:
            return False, "Feature order mismatch:\n" + "\n".join(mismatches[:5])
        
        return True, "Feature order validated successfully"
    
    def validate_feature_types(self, features: np.ndarray) -> Tuple[bool, str]:
        """
        Validate feature data types and values
        
        Args:
            features: Feature array to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for NaN values
        if np.any(np.isnan(features)):
            nan_cols = np.where(np.isnan(features).any(axis=0))[0]
            return False, f"Features contain NaN values in columns: {nan_cols.tolist()}"
        
        # Check for infinite values
        if np.any(np.isinf(features)):
            inf_cols = np.where(np.isinf(features).any(axis=0))[0]
            return False, f"Features contain infinite values in columns: {inf_cols.tolist()}"
        
        # Check data type
        if not np.issubdtype(features.dtype, np.number):
            return False, f"Features must be numeric, got dtype: {features.dtype}"
        
        return True, "Feature types validated successfully"
    
    def detect_feature_drift(
        self, 
        features: np.ndarray, 
        threshold: float = 3.0
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Detect feature drift by comparing against training statistics
        
        Args:
            features: Feature array to check for drift
            threshold: Number of standard deviations for drift detection
            
        Returns:
            Tuple of (has_drift, drift_report)
        """
        if not self.metadata or 'feature_stats' not in self.metadata:
            return False, {"warning": "No feature statistics available for drift detection"}
        
        stats = self.metadata['feature_stats']
        feature_names = self.metadata.get('feature_names', [])
        
        drift_report = {
            'drifted_features': [],
            'warnings': [],
            'feature_details': {}
        }
        
        has_drift = False
        
        for i in range(features.shape[1]):
            if i >= len(feature_names):
                break
            
            feature_name = feature_names[i]
            feature_value = features[0, i]
            
            if feature_name not in stats:
                continue
            
            stat = stats[feature_name]
            mean = stat.get('mean', 0)
            std = stat.get('std', 1)
            min_val = stat.get('min', -np.inf)
            max_val = stat.get('max', np.inf)
            
            # Calculate z-score
            z_score = abs((feature_value - mean) / std) if std > 0 else 0
            
            # Check if outside training range
            outside_range = feature_value < min_val or feature_value > max_val
            
            # Check if beyond threshold
            beyond_threshold = z_score > threshold
            
            drift_report['feature_details'][feature_name] = {
                'value': float(feature_value),
                'mean': mean,
                'std': std,
                'z_score': float(z_score),
                'outside_range': outside_range,
                'beyond_threshold': beyond_threshold
            }
            
            if beyond_threshold:
                has_drift = True
                drift_report['drifted_features'].append({
                    'name': feature_name,
                    'value': float(feature_value),
                    'z_score': float(z_score),
                    'expected_range': f"[{mean - threshold*std:.4f}, {mean + threshold*std:.4f}]"
                })
            
            if outside_range:
                drift_report['warnings'].append(
                    f"{feature_name}: {feature_value:.4f} outside training range [{min_val:.4f}, {max_val:.4f}]"
                )
        
        if has_drift:
            logger.warning(f"Feature drift detected in {len(drift_report['drifted_features'])} features")
        else:
            logger.info("No significant feature drift detected")
        
        return has_drift, drift_report
    
    def validate_all(
        self, 
        features: np.ndarray, 
        feature_names: Optional[list] = None,
        check_drift: bool = True,
        drift_threshold: float = 3.0
    ) -> Dict[str, any]:
        """
        Perform comprehensive feature validation
        
        Args:
            features: Feature array to validate
            feature_names: Optional list of feature names
            check_drift: Whether to check for feature drift
            drift_threshold: Z-score threshold for drift detection
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'drift_report': None
        }
        
        # Validate feature types
        types_valid, types_msg = self.validate_feature_types(features)
        if not types_valid:
            results['valid'] = False
            results['errors'].append(types_msg)
            return results
        
        # Validate feature order
        if feature_names:
            order_valid, order_msg = self.validate_feature_order(feature_names)
            if not order_valid:
                results['valid'] = False
                results['errors'].append(order_msg)
                return results
        
        # Detect feature drift
        if check_drift:
            has_drift, drift_report = self.detect_feature_drift(features, drift_threshold)
            results['drift_report'] = drift_report
            
            if has_drift:
                results['warnings'].append(
                    f"Feature drift detected in {len(drift_report['drifted_features'])} features"
                )
            
            results['warnings'].extend(drift_report.get('warnings', []))
        
        if results['valid']:
            logger.info("All feature validations passed")
        else:
            logger.error(f"Feature validation failed: {results['errors']}")
        
        return results
    
    @staticmethod
    def save_feature_metadata(
        feature_names: list,
        feature_stats: Dict[str, Dict],
        model_config: Dict,
        output_path: str = 'artifacts/feature_metadata.json'
    ):
        """
        Save feature metadata for future validation
        
        Args:
            feature_names: List of feature names in order
            feature_stats: Dictionary of feature statistics (mean, std, min, max)
            model_config: Model configuration dictionary
            output_path: Path to save metadata
        """
        metadata = {
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'feature_stats': feature_stats,
            'model_config': model_config,
            'version': '1.0'
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved feature metadata to {output_path}")
    
    @staticmethod
    def calculate_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate statistics for all features in a DataFrame
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }
        
        logger.info(f"Calculated statistics for {len(stats)} features")
        return stats
