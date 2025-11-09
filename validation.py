"""
Validation Framework for Mat-Su Gold Prospectivity System
==========================================================
Validates model predictions against known mine locations.
"""

import ee
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import config

class ValidationFramework:
    """
    Validates prospectivity model against known mine locations.

    The validation ensures that >= 80% of known successful mines
    fall within "High" prospectivity zones.
    """

    def __init__(self, user_mines_csv=None):
        """
        Initialize validation framework.

        Args:
            user_mines_csv: Path to CSV file with user's mine locations
                           Expected columns: name, latitude, longitude, type
        """
        self.config = config
        self.known_mines = self._load_known_mines(user_mines_csv)
        self.validation_results = None

    def _load_known_mines(self, user_csv=None):
        """
        Load known mine locations from config or user CSV.

        Args:
            user_csv: Optional path to user's CSV file

        Returns:
            pandas.DataFrame with columns: name, lat, lon, type
        """
        if user_csv and Path(user_csv).exists():
            print(f"✓ Loading user mine data from: {user_csv}")
            df = pd.read_csv(user_csv)
            # Standardize column names
            df.columns = df.columns.str.lower()
            if 'latitude' in df.columns:
                df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
            return df
        else:
            print("⚠ No user CSV provided, using config known mines")
            # Convert config KNOWN_MINES to DataFrame
            mines_data = []
            for name, info in config.KNOWN_MINES.items():
                mines_data.append({
                    'name': name,
                    'lat': info['lat'],
                    'lon': info['lon'],
                    'type': info['type']
                })
            return pd.DataFrame(mines_data)

    def sample_prospectivity_at_mines(self, prospectivity_image, scale=30):
        """
        Sample prospectivity values at known mine locations.

        Args:
            prospectivity_image: ee.Image - Prospectivity map (0-1 scale)
            scale: int - Sampling resolution in meters

        Returns:
            pandas.DataFrame with mine locations and prospectivity scores
        """
        print(f"\nSampling prospectivity at {len(self.known_mines)} mine locations...")

        results = []

        for idx, mine in self.known_mines.iterrows():
            name = mine['name']
            lat = mine['lat']
            lon = mine['lon']
            mine_type = mine.get('type', 'unknown')

            # Create point geometry
            point = ee.Geometry.Point([lon, lat])

            # Sample prospectivity at this point
            try:
                sample = prospectivity_image.sample(
                    region=point,
                    scale=scale,
                    geometries=True
                ).first().getInfo()

                if sample and 'properties' in sample:
                    # Get prospectivity value (handle different band names)
                    props = sample['properties']
                    prospectivity_value = None

                    # Try common band names
                    for key in ['prospectivity', 'constant', 'classification']:
                        if key in props:
                            prospectivity_value = props[key]
                            break

                    if prospectivity_value is None:
                        # Use first available numeric value
                        prospectivity_value = next((v for v in props.values() if isinstance(v, (int, float))), None)

                    results.append({
                        'name': name,
                        'lat': lat,
                        'lon': lon,
                        'type': mine_type,
                        'prospectivity': prospectivity_value,
                        'classification': self._classify_prospectivity(prospectivity_value)
                    })
                    print(f"  ✓ {name}: {prospectivity_value:.3f} ({self._classify_prospectivity(prospectivity_value)})")
                else:
                    print(f"  ✗ {name}: No data available at location")
                    results.append({
                        'name': name,
                        'lat': lat,
                        'lon': lon,
                        'type': mine_type,
                        'prospectivity': None,
                        'classification': 'NO_DATA'
                    })
            except Exception as e:
                print(f"  ✗ {name}: Error sampling - {str(e)}")
                results.append({
                    'name': name,
                    'lat': lat,
                    'lon': lon,
                    'type': mine_type,
                    'prospectivity': None,
                    'classification': 'ERROR'
                })

        return pd.DataFrame(results)

    def _classify_prospectivity(self, value):
        """
        Classify prospectivity value into categories.

        Args:
            value: float - Prospectivity score (0-1)

        Returns:
            str - Classification (HIGH/MEDIUM/LOW/VERY_LOW)
        """
        if value is None:
            return 'NO_DATA'

        thresholds = config.PROSPECTIVITY_THRESHOLDS
        if value >= thresholds['high']:
            return 'HIGH'
        elif value >= thresholds['medium']:
            return 'MEDIUM'
        elif value >= thresholds['low']:
            return 'LOW'
        else:
            return 'VERY_LOW'

    def validate_model(self, prospectivity_image, scale=30):
        """
        Complete validation of prospectivity model.

        Args:
            prospectivity_image: ee.Image - Prospectivity map
            scale: int - Sampling resolution

        Returns:
            dict - Validation metrics and results
        """
        print("\n" + "=" * 70)
        print("MODEL VALIDATION")
        print("=" * 70)

        # Sample prospectivity at mine locations
        self.validation_results = self.sample_prospectivity_at_mines(prospectivity_image, scale)

        # Calculate metrics
        metrics = self._calculate_validation_metrics()

        # Print results
        self._print_validation_report(metrics)

        # Save results
        self._save_validation_results(metrics)

        return metrics

    def _calculate_validation_metrics(self):
        """Calculate validation metrics."""
        df = self.validation_results

        # Filter out NO_DATA and ERROR entries
        valid_df = df[~df['classification'].isin(['NO_DATA', 'ERROR'])]

        if len(valid_df) == 0:
            return {
                'total_mines': len(df),
                'valid_samples': 0,
                'detection_rate': 0.0,
                'passed': False,
                'message': 'No valid samples obtained'
            }

        # Count detections
        high_count = len(valid_df[valid_df['classification'] == 'HIGH'])
        medium_count = len(valid_df[valid_df['classification'] == 'MEDIUM'])
        low_count = len(valid_df[valid_df['classification'] == 'LOW'])
        very_low_count = len(valid_df[valid_df['classification'] == 'VERY_LOW'])

        # Detection rate (HIGH zones only)
        detection_rate = high_count / len(valid_df)

        # Check if validation passed
        min_required = config.VALIDATION_CRITERIA['min_detection_rate']
        passed = detection_rate >= min_required

        metrics = {
            'total_mines': len(df),
            'valid_samples': len(valid_df),
            'high_count': high_count,
            'medium_count': medium_count,
            'low_count': low_count,
            'very_low_count': very_low_count,
            'detection_rate': detection_rate,
            'required_rate': min_required,
            'passed': passed,
            'mean_prospectivity': valid_df['prospectivity'].mean(),
            'median_prospectivity': valid_df['prospectivity'].median(),
            'std_prospectivity': valid_df['prospectivity'].std(),
        }

        return metrics

    def _print_validation_report(self, metrics):
        """Print validation report."""
        print(f"\nTotal Mines: {metrics['total_mines']}")
        print(f"Valid Samples: {metrics['valid_samples']}")
        print(f"\nClassification Breakdown:")
        print(f"  HIGH:      {metrics['high_count']:3d} ({metrics['high_count']/metrics['valid_samples']*100:5.1f}%)")
        print(f"  MEDIUM:    {metrics['medium_count']:3d} ({metrics['medium_count']/metrics['valid_samples']*100:5.1f}%)")
        print(f"  LOW:       {metrics['low_count']:3d} ({metrics['low_count']/metrics['valid_samples']*100:5.1f}%)")
        print(f"  VERY_LOW:  {metrics['very_low_count']:3d} ({metrics['very_low_count']/metrics['valid_samples']*100:5.1f}%)")

        print(f"\nProspectivity Statistics:")
        print(f"  Mean:   {metrics['mean_prospectivity']:.3f}")
        print(f"  Median: {metrics['median_prospectivity']:.3f}")
        print(f"  Std:    {metrics['std_prospectivity']:.3f}")

        print(f"\n{'=' * 70}")
        print(f"DETECTION RATE: {metrics['detection_rate']*100:.1f}% (Required: {metrics['required_rate']*100:.1f}%)")
        print(f"{'=' * 70}")

        if metrics['passed']:
            print("✓ VALIDATION PASSED!")
            print(f"  {metrics['high_count']} out of {metrics['valid_samples']} known mines detected in HIGH zones")
        else:
            print("✗ VALIDATION FAILED!")
            print(f"  Only {metrics['high_count']} out of {metrics['valid_samples']} mines in HIGH zones")
            print(f"  Need at least {int(metrics['required_rate']*metrics['valid_samples'])} mines in HIGH zones")
            print("\nRECOMMENDATIONS:")
            print("  1. Adjust prospectivity weights in config.py")
            print("  2. Tune ML hyperparameters")
            print("  3. Review normalization parameters")
            print("  4. Check for data quality issues at failed mine locations")

        print("=" * 70)

    def _save_validation_results(self, metrics):
        """Save validation results to files."""
        output_dir = config.VALIDATION_DIR

        # Save detailed results CSV
        csv_path = output_dir / f"{config.OUTPUT_PREFIX}_validation_results.csv"
        self.validation_results.to_csv(csv_path, index=False)
        print(f"\n✓ Validation results saved: {csv_path}")

        # Save metrics JSON
        json_path = output_dir / f"{config.OUTPUT_PREFIX}_validation_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Validation metrics saved: {json_path}")

    def get_failed_detections(self):
        """
        Get list of mines that failed to be detected in HIGH zones.

        Returns:
            pandas.DataFrame - Mines not in HIGH zones
        """
        if self.validation_results is None:
            print("⚠ No validation results available. Run validate_model() first.")
            return None

        failed = self.validation_results[
            (self.validation_results['classification'] != 'HIGH') &
            (~self.validation_results['classification'].isin(['NO_DATA', 'ERROR']))
        ]

        return failed

    def suggest_improvements(self):
        """
        Analyze failed detections and suggest improvements.
        """
        failed = self.get_failed_detections()

        if failed is None or len(failed) == 0:
            print("✓ All mines detected in HIGH zones - no improvements needed!")
            return

        print("\n" + "=" * 70)
        print("IMPROVEMENT SUGGESTIONS")
        print("=" * 70)
        print(f"\nFailed to detect {len(failed)} mines in HIGH zones:")

        for idx, mine in failed.iterrows():
            print(f"\n  {mine['name']}:")
            print(f"    Location: {mine['lat']:.4f}, {mine['lon']:.4f}")
            print(f"    Prospectivity: {mine['prospectivity']:.3f}")
            print(f"    Classification: {mine['classification']}")
            print(f"    Gap to HIGH: {config.PROSPECTIVITY_THRESHOLDS['high'] - mine['prospectivity']:.3f}")

        print(f"\n{'=' * 70}")
        print("ACTION ITEMS:")
        print("  1. Visually inspect failed mine locations in the map")
        print("  2. Check if spectral indices are detecting alteration at these sites")
        print("  3. Verify structural features (faults) near these mines")
        print("  4. Consider adjusting weights in config.PROSPECTIVITY_WEIGHTS")
        print("  5. Review training data to ensure similar signatures are included")
        print("=" * 70)


def quick_validation_check(prospectivity_image, user_mines_csv=None):
    """
    Quick validation check function.

    Args:
        prospectivity_image: ee.Image - Prospectivity map
        user_mines_csv: Optional path to user's mine CSV

    Returns:
        bool - True if validation passed
    """
    validator = ValidationFramework(user_mines_csv)
    metrics = validator.validate_model(prospectivity_image)
    return metrics['passed']


if __name__ == "__main__":
    # Example usage
    print("Validation Framework Module")
    print("=" * 70)
    print("\nThis module validates prospectivity models against known mines.")
    print("\nUsage:")
    print("  from validation import ValidationFramework")
    print("  validator = ValidationFramework('path/to/mines.csv')")
    print("  metrics = validator.validate_model(prospectivity_image)")
    print("\nExpected CSV format:")
    print("  name,latitude,longitude,type")
    print("  Independence Mine,61.7817,-149.2819,historic_producer")
    print("  Lucky Shot Mine,61.7750,-149.2889,historic_producer")
    print("=" * 70)
