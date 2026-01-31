"""
Feature Extraction Script for IMU Sensor Data
==============================================
This script processes IMU sensor data from multiple participants and walking types,
extracts statistical features, and creates a machine learning-ready dataset.

Author: Generated for IMU Walking Analysis
Date: 2025
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# Path configurations
RAW_DATA_PATH = "Raw Data"
OUTPUT_CSV_PATH = "ml_data.csv"
CORRELATION_PNG_PATH = "feature_correlation.png"
CORRELATION_MD_PATH = "feature_correlation.md"

# Walking type labels
WALK_TYPES = ["FROG", "HEEL", "KANGAROO", "SCISSOR", "SIDEKICK", "TIPTOE"]

# Columns to extract from IMU data (excluding DateTime, Timestamp, and Calibration columns)
IMU_COLUMNS = [
    'Gyro_X', 'Gyro_Y', 'Gyro_Z',
    'LinearAccel_X', 'LinearAccel_Y', 'LinearAccel_Z',
    'Magnetometer_X', 'Magnetometer_Y', 'Magnetometer_Z',
    'RawAccel_X', 'RawAccel_Y', 'RawAccel_Z',
    'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z'
]

# Feature names (12 features per column)
FEATURE_NAMES = [
    'mean', 'std', 'min', 'max', 'median', 'iqr',
    'skewness', 'kurtosis', 'rms', 'mad', 'log_detector', 'linear_slope'
]

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log_info(message):
    """Print INFO level log message"""
    print(f"[INFO] {message}")

def log_warning(message):
    """Print WARNING level log message"""
    print(f"[WARNING] {message}")

def log_error(message):
    """Print ERROR level log message"""
    print(f"[ERROR] {message}")

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def compute_mean(data):
    """Compute mean of the signal"""
    return np.mean(data)

def compute_std(data):
    """Compute standard deviation of the signal"""
    return np.std(data)

def compute_min(data):
    """Compute minimum value of the signal"""
    return np.min(data)

def compute_max(data):
    """Compute maximum value of the signal"""
    return np.max(data)

def compute_median(data):
    """Compute median of the signal"""
    return np.median(data)

def compute_iqr(data):
    """Compute interquartile range (IQR) of the signal"""
    q75, q25 = np.percentile(data, [75, 25])
    return q75 - q25

def compute_skewness(data):
    """Compute skewness of the signal"""
    return skew(data)

def compute_kurtosis(data):
    """Compute kurtosis of the signal"""
    return kurtosis(data)

def compute_rms(data):
    """Compute root mean square (RMS) of the signal"""
    return np.sqrt(np.mean(data ** 2))

def compute_mad(data):
    """Compute mean absolute deviation (MAD) of the signal"""
    return np.mean(np.abs(data - np.mean(data)))

def compute_log_detector(data):
    """
    Compute log detector (geometric mean of absolute values)
    Formula: exp(mean(ln(|x|)))
    """
    abs_data = np.abs(data)
    # Avoid log(0) by adding small epsilon
    abs_data = abs_data + 1e-10
    return np.exp(np.mean(np.log(abs_data)))

def compute_linear_slope(data):
    """Compute slope of linear fit to the signal"""
    x = np.arange(len(data))
    slope, _ = np.polyfit(x, data, 1)
    return slope

def extract_features_from_column(column_data, column_name):
    """
    Extract all statistical features from a single column
    
    Parameters:
    -----------
    column_data : array-like
        Data from a single IMU column
    column_name : str
        Name of the column
    
    Returns:
    --------
    dict : Dictionary of feature_name: feature_value
    """
    features = {}
    
    # Compute all features
    features[f'{column_name}_mean'] = compute_mean(column_data)
    features[f'{column_name}_std'] = compute_std(column_data)
    features[f'{column_name}_min'] = compute_min(column_data)
    features[f'{column_name}_max'] = compute_max(column_data)
    features[f'{column_name}_median'] = compute_median(column_data)
    features[f'{column_name}_iqr'] = compute_iqr(column_data)
    features[f'{column_name}_skewness'] = compute_skewness(column_data)
    features[f'{column_name}_kurtosis'] = compute_kurtosis(column_data)
    features[f'{column_name}_rms'] = compute_rms(column_data)
    features[f'{column_name}_mad'] = compute_mad(column_data)
    features[f'{column_name}_log_detector'] = compute_log_detector(column_data)
    features[f'{column_name}_linear_slope'] = compute_linear_slope(column_data)
    
    return features

def extract_features_from_csv(csv_path, walk_type):
    """
    Extract all features from a single CSV file
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    walk_type : str
        Type of walk (label)
    
    Returns:
    --------
    dict : Dictionary containing 'label' and all extracted features
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        missing_cols = [col for col in IMU_COLUMNS if col not in df.columns]
        if missing_cols:
            log_warning(f"Missing columns in {csv_path}: {missing_cols}")
            return None
        
        # Initialize feature dictionary
        features = {'label': walk_type}
        
        # Extract features from each IMU column
        for column in IMU_COLUMNS:
            column_data = df[column].values
            column_features = extract_features_from_column(column_data, column)
            features.update(column_features)
        
        log_info(f"Successfully extracted features from {os.path.basename(csv_path)}")
        return features
        
    except Exception as e:
        log_error(f"Error processing {csv_path}: {str(e)}")
        return None

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def find_csv_files_for_walk_type(participant_folder, walk_type):
    """
    Find CSV file matching the walk type in participant folder
    
    Parameters:
    -----------
    participant_folder : str
        Path to participant's folder
    walk_type : str
        Type of walk to search for
    
    Returns:
    --------
    str or None : Path to the CSV file, or None if not found
    """
    try:
        files = os.listdir(participant_folder)
        for file in files:
            if file.upper().find(walk_type.upper()) != -1 and file.endswith('.csv'):
                return os.path.join(participant_folder, file)
        return None
    except Exception as e:
        log_error(f"Error searching folder {participant_folder}: {str(e)}")
        return None

def process_all_participants():
    """
    Process all participants and extract features from all walking types
    
    Returns:
    --------
    list : List of dictionaries, each containing features for one CSV file
    """
    all_features = []
    
    # Check if raw data path exists
    if not os.path.exists(RAW_DATA_PATH):
        log_error(f"Raw data path does not exist: {RAW_DATA_PATH}")
        return all_features
    
    # Get all participant folders
    participant_folders = [f for f in os.listdir(RAW_DATA_PATH) 
                          if os.path.isdir(os.path.join(RAW_DATA_PATH, f))]
    participant_folders.sort()
    
    log_info(f"Found {len(participant_folders)} participant folders")
    
    # Process each participant
    for participant_folder in participant_folders:
        participant_path = os.path.join(RAW_DATA_PATH, participant_folder)
        log_info(f"Processing participant: {participant_folder}")
        
        # Process each walk type
        for walk_type in WALK_TYPES:
            csv_file = find_csv_files_for_walk_type(participant_path, walk_type)
            
            if csv_file is None:
                log_warning(f"  {walk_type} walk type not found for {participant_folder}")
                continue
            
            # Extract features
            features = extract_features_from_csv(csv_file, walk_type)
            
            if features is not None:
                all_features.append(features)
                log_info(f"  ✓ Processed {walk_type} - Total samples: {len(all_features)}")
            else:
                log_error(f"  ✗ Failed to process {walk_type}")
    
    log_info(f"\nTotal samples collected: {len(all_features)}")
    return all_features

def create_ml_dataset(all_features):
    """
    Create machine learning dataset from extracted features
    
    Parameters:
    -----------
    all_features : list
        List of feature dictionaries
    
    Returns:
    --------
    DataFrame : Pandas DataFrame with label and all features
    """
    if not all_features:
        log_error("No features to create dataset")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns to have 'label' first
    cols = ['label'] + [col for col in df.columns if col != 'label']
    df = df[cols]
    
    log_info(f"Created dataset with shape: {df.shape}")
    log_info(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df

# ============================================================================
# FEATURE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_feature_correlation(df, output_png, output_md):
    """
    Analyze and visualize feature correlations
    
    Parameters:
    -----------
    df : DataFrame
        Machine learning dataset
    output_png : str
        Path to save correlation heatmap
    output_md : str
        Path to save correlation analysis markdown
    """
    log_info("Performing feature correlation analysis...")
    
    # Get feature columns (exclude label)
    feature_cols = [col for col in df.columns if col != 'label']
    
    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Find highly correlated feature pairs (|correlation| > 0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    # Create heatmap
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Correlation heatmap saved to {output_png}")
    
    # Create markdown report
    with open(output_md, 'w') as f:
        f.write("# Feature Correlation Analysis\n\n")
        f.write(f"**Total Features:** {len(feature_cols)}\n\n")
        f.write(f"**Dataset Shape:** {df.shape}\n\n")
        
        f.write("## Highly Correlated Feature Pairs (|r| > 0.9)\n\n")
        if high_corr_pairs:
            f.write("| Feature 1 | Feature 2 | Correlation |\n")
            f.write("|-----------|-----------|-------------|\n")
            for pair in high_corr_pairs:
                f.write(f"| {pair['Feature 1']} | {pair['Feature 2']} | {pair['Correlation']:.4f} |\n")
        else:
            f.write("No highly correlated feature pairs found.\n")
        
        f.write("\n## Feature Statistics\n\n")
        f.write("| Statistic | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Mean Correlation | {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.4f} |\n")
        f.write(f"| Max Correlation | {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.4f} |\n")
        f.write(f"| Min Correlation | {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.4f} |\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("- Features with correlation > 0.9 may be redundant\n")
        f.write("- Consider feature selection techniques in the ML pipeline\n")
        f.write("- Use dimensionality reduction (PCA) if needed\n")
    
    log_info(f"Correlation analysis saved to {output_md}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("IMU SENSOR DATA - FEATURE EXTRACTION")
    print("="*80)
    print()
    
    # Step 1: Process all participants and extract features
    log_info("Starting feature extraction process...")
    all_features = process_all_participants()
    
    if not all_features:
        log_error("No features extracted. Exiting.")
        return
    
    # Step 2: Create ML dataset
    log_info("\nCreating machine learning dataset...")
    ml_dataset = create_ml_dataset(all_features)
    
    if ml_dataset is None:
        log_error("Failed to create dataset. Exiting.")
        return
    
    # Step 3: Save dataset to CSV
    log_info(f"\nSaving dataset to {OUTPUT_CSV_PATH}...")
    ml_dataset.to_csv(OUTPUT_CSV_PATH, index=False)
    log_info(f"✓ Dataset saved successfully!")
    
    # Step 4: Perform correlation analysis
    log_info("\nPerforming feature correlation analysis...")
    analyze_feature_correlation(ml_dataset, CORRELATION_PNG_PATH, CORRELATION_MD_PATH)
    
    # Final summary
    print()
    print("="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"Output Files:")
    print(f"  - {OUTPUT_CSV_PATH}")
    print(f"  - {CORRELATION_PNG_PATH}")
    print(f"  - {CORRELATION_MD_PATH}")
    print()
    print(f"Dataset Summary:")
    print(f"  - Total Samples: {len(ml_dataset)}")
    print(f"  - Total Features: {len(ml_dataset.columns) - 1}")
    print(f"  - Labels: {', '.join(WALK_TYPES)}")
    print("="*80)

if __name__ == "__main__":
    main()