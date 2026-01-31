"""
Machine Learning Classification Script for IMU Walking Type Prediction
=======================================================================
This script performs multi-class classification on IMU sensor features using
various supervised learning algorithms and generates comprehensive evaluation metrics.

Author: Generated for IMU Walking Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# File paths
INPUT_CSV_PATH = "ml_data_augmented.csv"  # Using augmented dataset with participant-based split
MODEL_ACCURACY_CSV = "model_accuracy_table.csv"
CLASSIFICATION_REPORT_MD = "classification_report.md"
CONFUSION_MATRIX_DIR = "confusion_matrices"
ROC_CURVE_DIR = "roc_curves"

# Train-test split ratio
TEST_SIZE = 0.35
TRAIN_SIZE = 0.65
RANDOM_STATE = 42

# Split method: 'participant' or 'stratified'
SPLIT_METHOD = 'participant'  # Change to 'stratified' for random stratified split
AUGMENTATION_FACTOR = 10  # Must match the augmentation factor used in A2_data_augmentation.py
SAMPLES_PER_PARTICIPANT_ORIGINAL = 6  # 6 walking types per participant

# Model configurations
MODELS = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'Decision_Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=RANDOM_STATE),
    'SVM': SVC(probability=True, random_state=RANDOM_STATE),
    'Naive_Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

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
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(file_path):
    """
    Load machine learning dataset from CSV
    
    Parameters:
    -----------
    file_path : str
        Path to the ML dataset CSV file
    
    Returns:
    --------
    DataFrame : Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        log_info(f"Dataset loaded successfully from {file_path}")
        log_info(f"Dataset shape: {df.shape}")
        log_info(f"Labels: {df['label'].unique()}")
        return df
    except Exception as e:
        log_error(f"Error loading dataset: {str(e)}")
        return None

def prepare_data(df):
    """
    Prepare data for machine learning (split features and labels)
    
    Parameters:
    -----------
    df : DataFrame
        Input dataset
    
    Returns:
    --------
    tuple : (X, y) where X is features and y is labels
    """
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    log_info(f"Features shape: {X.shape}")
    log_info(f"Labels shape: {y.shape}")
    log_info(f"Label distribution:\n{y.value_counts()}")
    
    return X, y

def stratified_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Perform stratified train-test split to ensure equal label distribution
    WARNING: This ignores participant grouping and may cause data leakage!
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Label vector
    test_size : float
        Proportion of dataset for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    log_info(f"\nTrain-Test Split (Stratified - Random):")
    log_info(f"  WARNING: This method may include same participants in train and test sets!")
    log_info(f"  Training set size: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    log_info(f"  Test set size: {len(X_test)} ({test_size*100:.0f}%)")
    log_info(f"\nTraining set label distribution:\n{pd.Series(y_train).value_counts()}")
    log_info(f"\nTest set label distribution:\n{pd.Series(y_test).value_counts()}")
    
    return X_train, X_test, y_train, y_test

def participant_based_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, samples_per_participant=6):
    """
    Perform participant-based train-test split
    This ensures all samples from a participant go to either train OR test set,
    preventing data leakage and ensuring model generalization to unseen participants.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Label vector
    test_size : float
        Proportion of dataset for testing (applied to participants, not samples)
    random_state : int
        Random seed for reproducibility
    samples_per_participant : int
        Number of samples per participant (for augmented data: original_samples * augmentation_factor)
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    np.random.seed(random_state)
    
    # Calculate number of participants
    total_samples = len(X)
    num_participants = total_samples // samples_per_participant
    
    log_info(f"Participant-Based Split:")
    log_info(f"  Total samples: {total_samples}")
    log_info(f"  Samples per participant: {samples_per_participant}")
    log_info(f"  Total participants: {num_participants}")
    
    # Calculate how many participants for test
    num_test_participants = int(np.ceil(num_participants * test_size))
    num_train_participants = num_participants - num_test_participants
    
    log_info(f"  Training participants: {num_train_participants}")
    log_info(f"  Test participants: {num_test_participants}")
    
    # Create participant indices
    participant_indices = list(range(num_participants))
    
    # Randomly shuffle and split participants
    np.random.shuffle(participant_indices)
    train_participants = participant_indices[:num_train_participants]
    test_participants = participant_indices[num_train_participants:]
    
    # Convert participant indices to sample indices
    train_sample_indices = []
    test_sample_indices = []
    
    for p_idx in train_participants:
        start_idx = p_idx * samples_per_participant
        end_idx = start_idx + samples_per_participant
        train_sample_indices.extend(range(start_idx, end_idx))
    
    for p_idx in test_participants:
        start_idx = p_idx * samples_per_participant
        end_idx = start_idx + samples_per_participant
        test_sample_indices.extend(range(start_idx, end_idx))
    
    # Split the data
    X_train = X.iloc[train_sample_indices].reset_index(drop=True)
    X_test = X.iloc[test_sample_indices].reset_index(drop=True)
    y_train = y[train_sample_indices]
    y_test = y[test_sample_indices]
    
    log_info(f"\nTrain-Test Split (Participant-Based):")
    log_info(f"  Training set size: {len(X_train)} ({len(X_train)/total_samples*100:.1f}%)")
    log_info(f"  Test set size: {len(X_test)} ({len(X_test)/total_samples*100:.1f}%)")
    log_info(f"\nTraining set label distribution:\n{pd.Series(y_train).value_counts()}")
    log_info(f"\nTest set label distribution:\n{pd.Series(y_test).value_counts()}")
    log_info(f"\nTrain participants (0-indexed): {sorted(train_participants)}")
    log_info(f"Test participants (0-indexed): {sorted(test_participants)}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Test features
    
    Returns:
    --------
    tuple : (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_info("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_model(model, model_name, X_train, y_train):
    """
    Train a machine learning model
    
    Parameters:
    -----------
    model : sklearn model
        Model to train
    model_name : str
        Name of the model
    X_train : array
        Training features
    y_train : array
        Training labels
    
    Returns:
    --------
    model : Trained model
    """
    log_info(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    log_info(f"✓ {model_name} training complete")
    return model

def evaluate_model(model, model_name, X_test, y_test, label_encoder):
    """
    Evaluate model and return metrics
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    model_name : str
        Name of the model
    X_test : array
        Test features
    y_test : array
        Test labels
    label_encoder : LabelEncoder
        Label encoder for class names
    
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    log_info(f"{model_name} Accuracy: {accuracy:.4f}")
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name, class_names):
    """
    Plot and save confusion matrix
    
    Parameters:
    -----------
    y_test : array
        True labels
    y_pred : array
        Predicted labels
    model_name : str
        Name of the model
    class_names : list
        List of class names
    """
    import os
    os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(CONFUSION_MATRIX_DIR, f'confusionmat_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_info(f"Confusion matrix saved: {output_path}")

def plot_roc_curves(model_results, class_names, label_encoder):
    """
    Plot ROC curves for all models and classes
    
    Parameters:
    -----------
    model_results : list
        List of dictionaries containing model evaluation results
    class_names : list
        List of class names
    label_encoder : LabelEncoder
        Label encoder for class names
    """
    import os
    os.makedirs(ROC_CURVE_DIR, exist_ok=True)
    
    n_classes = len(class_names)
    
    for result in model_results:
        model_name = result['model_name']
        y_test = result['y_test']
        y_pred_proba = result['y_pred_proba']
        
        # Binarize labels for multi-class ROC using the range of encoded labels
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            # Check if this class exists in test set
            if np.sum(y_test_bin[:, i]) > 0:
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            else:
                # If class not in test set, use dummy values
                fpr[i] = np.array([0, 1])
                tpr[i] = np.array([0, 1])
                roc_auc[i] = 0.5
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            if roc_auc[i] != 0.5 or np.sum(y_test_bin[:, i]) > 0:
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
            else:
                plt.plot([], [], color=color, lw=2,
                        label=f'{class_names[i]} (Not in test set)')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {model_name}', fontsize=16, pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(ROC_CURVE_DIR, f'roc_curve_{model_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_info(f"ROC curve saved: {output_path}")

def plot_auc_comparison(model_results, class_names, label_encoder):
    """
    Plot AUC comparison across all models
    
    Parameters:
    -----------
    model_results : list
        List of dictionaries containing model evaluation results
    class_names : list
        List of class names
    label_encoder : LabelEncoder
        Label encoder for class names
    """
    import os
    os.makedirs(ROC_CURVE_DIR, exist_ok=True)
    
    n_classes = len(class_names)
    auc_data = []
    
    for result in model_results:
        model_name = result['model_name']
        y_test = result['y_test']
        y_pred_proba = result['y_pred_proba']
        
        # Binarize labels using the range of encoded labels
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # Calculate AUC for each class
        for i in range(n_classes):
            try:
                # Check if class exists in test set
                if np.sum(y_test_bin[:, i]) > 0:
                    auc_score = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
                    auc_data.append({
                        'Model': model_name,
                        'Class': class_names[i],
                        'AUC': auc_score
                    })
                else:
                    # Class not in test set
                    auc_data.append({
                        'Model': model_name,
                        'Class': class_names[i],
                        'AUC': np.nan
                    })
            except Exception as e:
                log_warning(f"Could not calculate AUC for {model_name} - {class_names[i]}: {str(e)}")
                auc_data.append({
                    'Model': model_name,
                    'Class': class_names[i],
                    'AUC': np.nan
                })
    
    # Create DataFrame and plot
    auc_df = pd.DataFrame(auc_data)
    
    plt.figure(figsize=(14, 8))
    pivot_df = auc_df.pivot(index='Model', columns='Class', values='AUC')
    
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'AUC Score'})
    plt.title('AUC Scores Across Models and Classes', fontsize=16, pad=20)
    plt.ylabel('Model', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(ROC_CURVE_DIR, 'auc_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_info(f"AUC comparison saved: {output_path}")

# ============================================================================
# REPORT GENERATION
# ============================================================================

def save_model_accuracy_table(model_results, output_path):
    """
    Save model accuracy comparison table
    
    Parameters:
    -----------
    model_results : list
        List of dictionaries containing model evaluation results
    output_path : str
        Path to save CSV file
    """
    results_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1-Score': r['f1_score']
    } for r in model_results])
    
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    results_df.to_csv(output_path, index=False)
    log_info(f"Model accuracy table saved: {output_path}")

def generate_classification_report_md(model_results, class_names, output_path):
    """
    Generate comprehensive classification report in Markdown
    
    Parameters:
    -----------
    model_results : list
        List of dictionaries containing model evaluation results
    class_names : list
        List of class names
    output_path : str
        Path to save markdown file
    """
    with open(output_path, 'w') as f:
        f.write("# Classification Report - IMU Walking Type Prediction\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Total Models Evaluated:** {len(model_results)}\n")
        f.write(f"- **Classes:** {', '.join(class_names)}\n")
        f.write(f"- **Train/Test Split:** {TRAIN_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%\n\n")
        
        f.write("## Model Performance Summary\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")
        
        # Sort by accuracy
        sorted_results = sorted(model_results, key=lambda x: x['accuracy'], reverse=True)
        
        for result in sorted_results:
            f.write(f"| {result['model_name']} | {result['accuracy']:.4f} | "
                   f"{result['precision']:.4f} | {result['recall']:.4f} | "
                   f"{result['f1_score']:.4f} |\n")
        
        f.write("\n## Detailed Classification Reports\n\n")
        
        for result in sorted_results:
            f.write(f"### {result['model_name']}\n\n")
            
            # Generate classification report
            report = classification_report(result['y_test'], result['y_pred'], 
                                          target_names=class_names, output_dict=True)
            
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|----------|\n")
            
            for class_name in class_names:
                metrics = report[class_name]
                f.write(f"| {class_name} | {metrics['precision']:.4f} | "
                       f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
                       f"{int(metrics['support'])} |\n")
            
            f.write(f"\n**Accuracy:** {report['accuracy']:.4f}\n")
            f.write(f"**Macro Avg:** Precision={report['macro avg']['precision']:.4f}, "
                   f"Recall={report['macro avg']['recall']:.4f}, "
                   f"F1-Score={report['macro avg']['f1-score']:.4f}\n")
            f.write(f"**Weighted Avg:** Precision={report['weighted avg']['precision']:.4f}, "
                   f"Recall={report['weighted avg']['recall']:.4f}, "
                   f"F1-Score={report['weighted avg']['f1-score']:.4f}\n\n")
            
            # Add sensitivity (same as recall for multi-class)
            f.write("**Sensitivity (Recall) by Class:**\n\n")
            for class_name in class_names:
                sensitivity = report[class_name]['recall']
                f.write(f"- {class_name}: {sensitivity:.4f}\n")
            f.write("\n---\n\n")
        
        f.write("## Interpretation Guide\n\n")
        f.write("- **Precision:** Proportion of positive predictions that are correct\n")
        f.write("- **Recall (Sensitivity):** Proportion of actual positives correctly identified\n")
        f.write("- **F1-Score:** Harmonic mean of precision and recall\n")
        f.write("- **Support:** Number of samples in each class\n")
    
    log_info(f"Classification report saved: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("MACHINE LEARNING CLASSIFICATION - IMU WALKING TYPE PREDICTION")
    print("="*80)
    print()
    
    # Step 1: Load data
    log_info("Loading dataset...")
    df = load_data(INPUT_CSV_PATH)
    if df is None:
        return
    
    # Step 2: Prepare data
    log_info("\nPreparing data...")
    X, y = prepare_data(df)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()
    
    # Step 3: Train-test split (choose method)
    log_info(f"\nSplitting data using '{SPLIT_METHOD}' method...")
    
    if SPLIT_METHOD == 'participant':
        # For augmented data, samples per participant = original_samples * augmentation_factor
        samples_per_participant = SAMPLES_PER_PARTICIPANT_ORIGINAL * AUGMENTATION_FACTOR
        X_train, X_test, y_train, y_test = participant_based_split(
            X, y_encoded, 
            samples_per_participant=samples_per_participant
        )
    elif SPLIT_METHOD == 'stratified':
        X_train, X_test, y_train, y_test = stratified_split(X, y_encoded)
    else:
        log_error(f"Unknown split method: {SPLIT_METHOD}. Use 'participant' or 'stratified'")
        return
    
    # Step 4: Scale features
    log_info("\nScaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 5: Train and evaluate all models
    log_info("\n" + "="*80)
    log_info("MODEL TRAINING AND EVALUATION")
    log_info("="*80)
    
    model_results = []
    
    for model_name, model in MODELS.items():
        # Train model
        trained_model = train_model(model, model_name, X_train_scaled, y_train)
        
        # Evaluate model
        results = evaluate_model(trained_model, model_name, X_test_scaled, 
                                y_test, label_encoder)
        model_results.append(results)
        
        # Plot confusion matrix
        plot_confusion_matrix(results['y_test'], results['y_pred'], 
                            model_name, class_names)
    
    # Step 6: Generate ROC curves
    log_info("\n" + "="*80)
    log_info("GENERATING ROC CURVES")
    log_info("="*80)
    plot_roc_curves(model_results, class_names, label_encoder)
    plot_auc_comparison(model_results, class_names, label_encoder)
    
    # Step 7: Save results
    log_info("\n" + "="*80)
    log_info("SAVING RESULTS")
    log_info("="*80)
    save_model_accuracy_table(model_results, MODEL_ACCURACY_CSV)
    generate_classification_report_md(model_results, class_names, CLASSIFICATION_REPORT_MD)
    
    # Final summary
    print()
    print("="*80)
    print("CLASSIFICATION COMPLETE")
    print("="*80)
    print(f"\nOutput Files Generated:")
    print(f"  - {MODEL_ACCURACY_CSV}")
    print(f"  - {CLASSIFICATION_REPORT_MD}")
    print(f"  - {CONFUSION_MATRIX_DIR}/ (confusion matrices)")
    print(f"  - {ROC_CURVE_DIR}/ (ROC curves and AUC comparison)")
    print()
    print("Best Performing Models:")
    sorted_results = sorted(model_results, key=lambda x: x['accuracy'], reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {result['model_name']}: {result['accuracy']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()