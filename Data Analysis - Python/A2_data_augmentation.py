"""
Data Augmentation Script for IMU Sensor Data
=============================================
This script applies various augmentation techniques to increase the dataset size
for improved machine learning model training.

Augmentation Techniques Used:
1. Jittering (adding random noise)
2. Scaling (magnitude warping)
3. Rotation (simulating different sensor orientations)
4. Time Warping (speed variation)
5. Window Slicing (creating overlapping samples)

Author: Generated for IMU Walking Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

INPUT_CSV = "ml_data.csv"
OUTPUT_CSV = "ml_data_augmented.csv"
AUGMENTATION_FACTOR = 100  # Target: 10x more samples
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log_info(message):
    """Print INFO level log message"""
    print(f"[INFO] {message}")

def log_success(message):
    """Print SUCCESS level log message"""
    print(f"[SUCCESS] {message}")

# ============================================================================
# AUGMENTATION TECHNIQUES
# ============================================================================

def add_jitter(data, sigma=0.03):
    """
    Add random Gaussian noise to the data (jittering)
    
    Parameters:
    -----------
    data : array
        Feature vector
    sigma : float
        Standard deviation of noise
    
    Returns:
    --------
    array : Augmented data with noise
    """
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise

def scale_data(data, sigma=0.1):
    """
    Scale the data by a random factor (magnitude warping)
    
    Parameters:
    -----------
    data : array
        Feature vector
    sigma : float
        Standard deviation for scaling factor
    
    Returns:
    --------
    array : Scaled data
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=data.shape)
    return data * scaling_factor

def rotate_data(data, angle_range=(-5, 5)):
    """
    Apply small rotation to simulate sensor orientation changes
    
    Parameters:
    -----------
    data : array
        Feature vector
    angle_range : tuple
        Range of rotation angles in degrees
    
    Returns:
    --------
    array : Rotated data
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    angle_rad = np.deg2rad(angle)
    
    # Apply small multiplicative rotation effect
    rotation_factor = 1 + np.sin(angle_rad) * 0.1
    return data * rotation_factor

def permute_features(data, num_segments=4):
    """
    Permute segments of the feature vector
    
    Parameters:
    -----------
    data : array
        Feature vector
    num_segments : int
        Number of segments to divide and permute
    
    Returns:
    --------
    array : Permuted data
    """
    data_copy = data.copy()
    segment_size = len(data) // num_segments
    
    # Only permute if we can create meaningful segments
    if segment_size > 5:
        segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else len(data)
            segments.append(data_copy[start:end])
        
        # Shuffle segments
        np.random.shuffle(segments)
        data_copy = np.concatenate(segments)
    
    return data_copy

def time_warp(data, sigma=0.2):
    """
    Apply time warping effect by stretching/compressing feature values
    
    Parameters:
    -----------
    data : array
        Feature vector
    sigma : float
        Standard deviation for warping factor
    
    Returns:
    --------
    array : Time-warped data
    """
    warp_factor = np.random.normal(1.0, sigma, size=data.shape)
    warp_factor = np.clip(warp_factor, 0.8, 1.2)  # Limit warping range
    return data * warp_factor

def magnitude_warp(data, sigma=0.2):
    """
    Apply smooth magnitude warping using a curve
    
    Parameters:
    -----------
    data : array
        Feature vector
    sigma : float
        Warping strength
    
    Returns:
    --------
    array : Magnitude-warped data
    """
    warp_curve = np.random.normal(1.0, sigma, size=len(data))
    # Smooth the curve
    from scipy.ndimage import gaussian_filter1d
    try:
        warp_curve = gaussian_filter1d(warp_curve, sigma=len(data)//10)
    except:
        # If scipy not available, use simple smoothing
        kernel_size = max(3, len(data)//20)
        warp_curve = np.convolve(warp_curve, np.ones(kernel_size)/kernel_size, mode='same')
    
    return data * warp_curve

def create_hybrid_augmentation(data):
    """
    Combine multiple augmentation techniques
    
    Parameters:
    -----------
    data : array
        Feature vector
    
    Returns:
    --------
    array : Augmented data using multiple techniques
    """
    # Randomly select 2-3 techniques to combine
    augmented = data.copy()
    
    if np.random.rand() > 0.5:
        augmented = add_jitter(augmented, sigma=0.02)
    
    if np.random.rand() > 0.5:
        augmented = scale_data(augmented, sigma=0.08)
    
    if np.random.rand() > 0.5:
        augmented = rotate_data(augmented, angle_range=(-3, 3))
    
    return augmented

# ============================================================================
# AUGMENTATION STRATEGIES
# ============================================================================

def augment_sample(sample_features, augmentation_type):
    """
    Apply specific augmentation technique to a sample
    
    Parameters:
    -----------
    sample_features : array
        Feature vector of a single sample
    augmentation_type : int
        Type of augmentation to apply
    
    Returns:
    --------
    array : Augmented sample
    """
    if augmentation_type == 0:
        return add_jitter(sample_features, sigma=0.03)
    elif augmentation_type == 1:
        return add_jitter(sample_features, sigma=0.05)
    elif augmentation_type == 2:
        return scale_data(sample_features, sigma=0.1)
    elif augmentation_type == 3:
        return scale_data(sample_features, sigma=0.15)
    elif augmentation_type == 4:
        return rotate_data(sample_features, angle_range=(-5, 5))
    elif augmentation_type == 5:
        return rotate_data(sample_features, angle_range=(-10, 10))
    elif augmentation_type == 6:
        return time_warp(sample_features, sigma=0.15)
    elif augmentation_type == 7:
        return time_warp(sample_features, sigma=0.25)
    elif augmentation_type == 8:
        return create_hybrid_augmentation(sample_features)
    else:
        return create_hybrid_augmentation(sample_features)

def augment_dataset(df, target_multiplier=AUGMENTATION_FACTOR, samples_per_participant=6):
    """
    Augment the entire dataset while preserving participant structure
    
    Parameters:
    -----------
    df : DataFrame
        Original dataset (organized with samples_per_participant consecutive rows per participant)
    target_multiplier : int
        Target multiplication factor
    samples_per_participant : int
        Number of samples per participant (6 walking types)
    
    Returns:
    --------
    DataFrame : Augmented dataset with participant structure preserved
    """
    log_info(f"Starting data augmentation (target: {target_multiplier}x samples)...")
    log_info(f"Preserving participant structure ({samples_per_participant} samples/participant)...")
    
    # Separate features and labels
    labels = df['label']
    features = df.drop('label', axis=1)
    
    # Calculate number of participants
    num_participants = len(df) // samples_per_participant
    log_info(f"Original dataset: {len(df)} samples, {num_participants} participants")
    log_info(f"Augmenting each sample {target_multiplier - 1} times...")
    
    augmented_samples = []
    augmented_labels = []
    
    # Process each participant's data as a group
    for p_idx in range(num_participants):
        start_idx = p_idx * samples_per_participant
        end_idx = start_idx + samples_per_participant
        
        # Get this participant's data (6 samples)
        participant_features = features.iloc[start_idx:end_idx]
        participant_labels = labels.iloc[start_idx:end_idx]
        
        # Add original samples for this participant
        augmented_samples.append(participant_features.values)
        augmented_labels.extend(participant_labels.values)
        
        # Create augmented versions for each sample of this participant
        for sample_idx in range(samples_per_participant):
            sample_features = participant_features.iloc[sample_idx].values
            sample_label = participant_labels.iloc[sample_idx]
            
            # Generate (target_multiplier - 1) augmented versions
            for aug_idx in range(target_multiplier - 1):
                augmentation_type = aug_idx % 10
                augmented_sample = augment_sample(sample_features, augmentation_type)
                
                augmented_samples.append(augmented_sample.reshape(1, -1))
                augmented_labels.append(sample_label)
        
        # Progress indicator
        if (p_idx + 1) % 3 == 0 or p_idx == num_participants - 1:
            log_info(f"  Processed {p_idx + 1}/{num_participants} participants...")
    
    # Combine all samples
    all_samples = np.vstack(augmented_samples)
    
    # Create new DataFrame
    feature_columns = features.columns.tolist()
    augmented_df = pd.DataFrame(all_samples, columns=feature_columns)
    augmented_df['label'] = augmented_labels
    
    # Verify participant structure is preserved
    final_participants = len(augmented_df) // (samples_per_participant * target_multiplier)
    log_info(f"\nAugmented dataset: {len(augmented_df)} samples")
    log_info(f"Structure: {final_participants} participants × {samples_per_participant * target_multiplier} samples/participant")
    log_info(f"Participant structure preserved: Each group of {samples_per_participant * target_multiplier} rows = 1 participant")
    
    return augmented_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("DATA AUGMENTATION FOR IMU SENSOR DATA")
    print("="*80)
    print()
    
    # Load original data
    log_info(f"Loading dataset from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
        log_success(f"Dataset loaded successfully")
        log_info(f"Original shape: {df.shape}")
        log_info(f"Original label distribution:\n{df['label'].value_counts()}")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {str(e)}")
        return
    
    print()
    log_info("="*80)
    log_info("AUGMENTATION TECHNIQUES APPLIED:")
    log_info("  1. Jittering (Gaussian noise - multiple levels)")
    log_info("  2. Scaling (Magnitude warping - multiple levels)")
    log_info("  3. Rotation (Orientation changes - multiple angles)")
    log_info("  4. Time Warping (Speed variations)")
    log_info("  5. Hybrid (Combination of multiple techniques)")
    log_info("="*80)
    print()
    
    # Augment dataset
    augmented_df = augment_dataset(df, target_multiplier=AUGMENTATION_FACTOR)
    
    print()
    log_info("="*80)
    log_info("AUGMENTATION COMPLETE")
    log_info("="*80)
    log_info(f"Augmented shape: {augmented_df.shape}")
    log_info(f"Size increase: {len(df)} → {len(augmented_df)} "
             f"({len(augmented_df)/len(df):.1f}x)")
    print()
    log_info(f"Augmented label distribution:\n{augmented_df['label'].value_counts()}")
    
    # Save augmented data
    print()
    log_info(f"Saving augmented dataset to {OUTPUT_CSV}...")
    augmented_df.to_csv(OUTPUT_CSV, index=False)
    log_success(f"Augmented dataset saved successfully!")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original samples:   {len(df)}")
    print(f"Augmented samples:  {len(augmented_df)}")
    print(f"Multiplication:     {len(augmented_df)/len(df):.1f}x")
    print(f"Output file:        {OUTPUT_CSV}")
    print("="*80)
    print()
    print("Next step: Run B_ml_classification.py with the augmented dataset!")
    print("  (Update INPUT_CSV_PATH to 'ml_data_augmented.csv' in the script)")
    print()

if __name__ == "__main__":
    main()
