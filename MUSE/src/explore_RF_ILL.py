"""
Random Forest Interpretability Analysis Functions for Input Loss Landscape (ILL) Features

This module provides comprehensive interpretability analysis functions for Random Forest models
trained on Input Loss Landscape features to distinguish between forget, retain, and holdout datasets.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP for advanced interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import pickle
import os


def prepare_classification_data(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor):
    """Prepare data for classification tasks"""
    
    # Combine all tensors
    datasets = {'retain': norm_retain_tensor, 'holdout': norm_holdout_tensor, 'forget': norm_forget_tensor}
    tensors = [tensor for tensor in datasets.values()]
    labels = [torch.full((len(tensor),), i) for i, tensor in enumerate(datasets.values())]
    
    X = torch.cat(tensors, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()
    
    # Create dataset names mapping
    class_names = list(datasets.keys())
    
    return X, y, class_names


def remove_features_and_evaluate(X_train, X_test, y_train, y_test, features_to_remove, feature_names, class_names):
    """
    Remove specified features and evaluate model performance
    """
    # Get indices of features to keep
    features_to_keep = [i for i, name in enumerate(feature_names) if name not in features_to_remove]
    
    # Select only the features to keep
    X_train_reduced = X_train[:, features_to_keep]
    X_test_reduced = X_test[:, features_to_keep]
    
    # Train Random Forest on reduced feature set
    rf_reduced = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    rf_reduced.fit(X_train_reduced, y_train)
    
    # Evaluate performance
    train_accuracy = rf_reduced.score(X_train_reduced, y_train)
    test_accuracy = rf_reduced.score(X_test_reduced, y_test)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_reduced, X_train_reduced, y_train, cv=5, scoring='accuracy')
    
    return {
        'model': rf_reduced,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'features_removed': features_to_remove,
        'features_kept': [feature_names[i] for i in features_to_keep],
        'n_features_kept': len(features_to_keep),
        'n_features_removed': len(features_to_remove)
    }


def ablation_study_bottom_k(k_values, feature_importance_df, X_train, X_test, y_train, y_test, feature_names, class_names):
    """
    Remove bottom k features (least important) and evaluate performance
    """
    results = {}
    
    print(f"Running ablation study: removing bottom k features...")
    
    for k in k_values:
        if k >= len(feature_names):
            print(f"Skipping k={k} (would remove all features)")
            continue
            
        # Get bottom k features (least important)
        bottom_features = feature_importance_df.tail(k)['feature'].tolist()
        
        result = remove_features_and_evaluate(
            X_train, X_test, y_train, y_test, bottom_features, feature_names, class_names
        )
        
        results[f'remove_bottom_{k}'] = result
        print(f"  Removed bottom {k} features: CV Accuracy = {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    
    return results


def ablation_study_top_k(k_values, feature_importance_df, X_train, X_test, y_train, y_test, feature_names, class_names):
    """
    Remove top k features (most important) and evaluate performance
    """
    results = {}
    
    print(f"Running ablation study: removing top k features...")
    
    for k in k_values:
        if k >= len(feature_names):
            print(f"Skipping k={k} (would remove all features)")
            continue
            
        # Get top k features (most important)
        top_features = feature_importance_df.head(k)['feature'].tolist()
        
        result = remove_features_and_evaluate(
            X_train, X_test, y_train, y_test, top_features, feature_names, class_names
        )
        
        results[f'remove_top_{k}'] = result
        print(f"  Removed top {k} features: CV Accuracy = {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    
    return results


def train_tree_surrogate(rf_model, X_train, X_test, y_train, y_test, feature_names, max_depth=None):
    """
    Train a decision tree to mimic the Random Forest's behavior
    """
    # Get Random Forest predictions on training data (this is what we want to mimic)
    rf_train_predictions = rf_model.predict(X_train)
    rf_test_predictions = rf_model.predict(X_test)
    
    # Train decision tree to predict what the RF predicts
    surrogate_tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    surrogate_tree.fit(X_train, rf_train_predictions)
    
    # Evaluate how well the tree mimics the RF
    tree_train_predictions = surrogate_tree.predict(X_train)
    tree_test_predictions = surrogate_tree.predict(X_test)
    
    # Fidelity: How well does the tree match the RF's predictions?
    train_fidelity = accuracy_score(rf_train_predictions, tree_train_predictions)
    test_fidelity = accuracy_score(rf_test_predictions, tree_test_predictions)
    
    # Accuracy: How well does the tree perform on the actual labels?
    train_accuracy = accuracy_score(y_train, tree_train_predictions)
    test_accuracy = accuracy_score(y_test, tree_test_predictions)
    
    return {
        'model': surrogate_tree,
        'train_fidelity': train_fidelity,
        'test_fidelity': test_fidelity,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'max_depth': max_depth,
        'n_leaves': surrogate_tree.get_n_leaves(),
        'tree_depth': surrogate_tree.get_depth()
    }


def run_permutation_importance_analysis(rf_model, X_test, y_test, feature_names):
    """
    Compute and analyze permutation importance
    """
    print("Computing Permutation Importance...")
    print("This measures how much performance drops when each feature is randomly shuffled")
    
    # Compute permutation importance
    perm_importance = permutation_importance(
        rf_model, X_test, y_test, 
        n_repeats=10, 
        random_state=42, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Create permutation importance dataframe
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return perm_importance_df, perm_importance


def analyze_partial_dependence(rf_model, X_test, feature_names, class_names, top_n_features=6):
    """
    Analyze partial dependence for top features
    """
    print("Computing Partial Dependence Plots...")
    print("These show how each feature individually affects model predictions")
    
    # Select top features for partial dependence analysis
    feature_importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    top_features_for_pdp = feature_importance_df.head(top_n_features)['feature'].tolist()
    
    pdp_insights = {}
    for feature in top_features_for_pdp:
        feature_idx = feature_names.index(feature)
        
        # Compute partial dependence
        pdp_result = partial_dependence(
            rf_model, X_test, features=[feature_idx], 
            grid_resolution=50, kind='average'
        )
        
        # Analyze the pattern for each class
        feature_insights = {}
        for class_idx, class_name in enumerate(class_names):
            pd_values = pdp_result['average'][class_idx]
            feature_values = pdp_result['grid_values'][0]
            
            # Calculate characteristics
            pd_range = pd_values.max() - pd_values.min()
            pd_mean = pd_values.mean()
            pd_std = pd_values.std()
            
            # Find optimal range (where this class has highest probability)
            max_prob_idx = np.argmax(pd_values)
            optimal_value = feature_values[max_prob_idx]
            max_prob = pd_values[max_prob_idx]
            
            feature_insights[class_name] = {
                'optimal_value': optimal_value,
                'max_probability': max_prob,
                'range': pd_range,
                'mean_effect': pd_mean,
                'variability': pd_std
            }
        
        pdp_insights[feature] = feature_insights
    
    return pdp_insights, top_features_for_pdp


def analyze_feature_interactions(rf_model, X_test, feature_names, top_n_features=8):
    """
    Analyze potential feature interactions using Random Forest
    """
    print("Analyzing potential feature interactions...")
    
    # Select top features for interaction analysis
    feature_importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    top_interaction_features = feature_importance_df.head(top_n_features)['feature'].tolist()
    
    # Compute 2D partial dependence for feature pairs
    interaction_strength = {}
    
    for i in range(len(top_interaction_features)):
        for j in range(i+1, len(top_interaction_features)):
            feature1 = top_interaction_features[i]
            feature2 = top_interaction_features[j]
            
            feature1_idx = feature_names.index(feature1)
            feature2_idx = feature_names.index(feature2)
            
            # Compute 2D partial dependence
            try:
                pdp_2d = partial_dependence(
                    rf_model, X_test, 
                    features=[feature1_idx, feature2_idx], 
                    grid_resolution=20, kind='average'
                )
                
                # Measure interaction strength as variance in the 2D surface
                interaction_strength[f"{feature1}_x_{feature2}"] = {
                    'features': (feature1, feature2),
                    'variance': np.var(pdp_2d['average']),
                    'range': np.ptp(pdp_2d['average'])  # peak-to-peak (max - min)
                }
                
            except Exception as e:
                print(f"Failed to compute interaction for {feature1} x {feature2}: {e}")
    
    # Sort interactions by strength
    sorted_interactions = []
    if interaction_strength:
        sorted_interactions = sorted(interaction_strength.items(), 
                                   key=lambda x: x[1]['variance'], reverse=True)
    
    return sorted_interactions


def run_shap_analysis(rf_model, X_test, y_test, feature_names, class_names, sample_size=100):
    """
    Run SHAP analysis if available
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return None, None
    
    print("Running SHAP Analysis...")
    print("This shows feature contributions for individual predictions")
    
    try:
        # Create SHAP explainer for Random Forest
        explainer = shap.TreeExplainer(rf_model)
        
        # Calculate SHAP values for a subset of test data (for efficiency)
        shap_sample_size = min(sample_size, len(X_test))
        X_test_sample = X_test[:shap_sample_size]
        y_test_sample = y_test[:shap_sample_size]
        
        print(f"Computing SHAP values for {shap_sample_size} test samples...")
        shap_values = explainer.shap_values(X_test_sample)
        
        # Feature importance from SHAP
        shap_importance = np.abs(shap_values).mean(axis=(0, 1))  # Average across samples and classes
        shap_importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        return shap_values, shap_importance_df, explainer, X_test_sample, y_test_sample
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None, None


def generate_comprehensive_summary(cv_scores, test_accuracy, feature_importance_df, perm_importance_df, 
                                 features_labels, rf_perm_corr=None, rf_tree_corr=None, 
                                 ablation_results=None, summary_df=None, best_surrogate=None,
                                 sorted_separation=None, sorted_interactions=None):
    """Generate a comprehensive summary of all interpretability analyses"""
    
    print("="*80)
    print("COMPREHENSIVE RANDOM FOREST INTERPRETABILITY SUMMARY")
    print("="*80)
    
    # 1. Model Performance Summary
    print("\n1. MODEL PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Baseline Random Forest Performance:")
    print(f"  Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Number of features used: {len(features_labels)}")
    
    # 2. Feature Importance Insights
    print("\n2. FEATURE IMPORTANCE INSIGHTS")
    print("-" * 40)
    print("Top 5 most important features (RF importance):")
    for i, row in feature_importance_df.head(5).iterrows():
        print(f"  {i + 1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\nTop 5 most important features (Permutation importance):")
    for i, row in perm_importance_df.head(5).iterrows():
        print(f"  {i + 1}. {row['feature']}: {row['importance_mean']:.4f}")
    
    # Feature importance agreement
    if rf_perm_corr is not None:
        print(f"\nFeature importance correlations:")
        print(f"  RF vs Permutation: {rf_perm_corr:.4f}")
        if rf_tree_corr is not None:
            print(f"  RF vs Tree Surrogate: {rf_tree_corr:.4f}")
        
        if rf_perm_corr > 0.7:
            print("  → High agreement between RF and permutation importance")
        elif rf_perm_corr > 0.4:
            print("  → Moderate agreement - some differences in feature ranking")
        else:
            print("  → Low agreement - significant differences in importance measures")
    
    # 3. Ablation Study Results
    if ablation_results is not None and summary_df is not None:
        print("\n3. FEATURE ABLATION INSIGHTS")
        print("-" * 40)
        
        if len(summary_df) > 1:
            best_ablation = summary_df[summary_df['experiment'] != 'baseline'].loc[
                summary_df[summary_df['experiment'] != 'baseline']['performance_drop'].idxmin()
            ]
            
            print(f"Best feature-reduced model:")
            print(f"  Experiment: {best_ablation['experiment']}")
            print(f"  Features kept: {best_ablation['features_kept']}/{len(features_labels)}")
            print(f"  Performance drop: {best_ablation['performance_drop']:.4f}")
            print(f"  CV Accuracy: {best_ablation['cv_mean']:.4f}")
    
    # 4. Tree Surrogate Insights
    if best_surrogate is not None:
        print("\n4. TREE SURROGATE MODEL INSIGHTS")
        print("-" * 40)
        print(f"Best surrogate tree performance:")
        print(f"  Tree depth: {best_surrogate['tree_depth']}")
        print(f"  Number of leaves: {best_surrogate['n_leaves']}")
        print(f"  Fidelity (mimics RF): {best_surrogate['test_fidelity']:.4f}")
        print(f"  Accuracy (actual): {best_surrogate['test_accuracy']:.4f}")
        
        if best_surrogate['test_fidelity'] > 0.8:
            print("  → High fidelity: Tree successfully captures RF behavior")
        elif best_surrogate['test_fidelity'] > 0.6:
            print("  → Moderate fidelity: Tree partially captures RF behavior")
        else:
            print("  → Low fidelity: Tree struggles to mimic RF")
    
    # 5. Partial Dependence Insights
    if sorted_separation is not None:
        print("\n5. PARTIAL DEPENDENCE INSIGHTS")
        print("-" * 40)
        print("Features with strongest class separation:")
        for i, (feature, scores) in enumerate(sorted_separation[:3]):
            print(f"  {i+1}. {feature}: separation = {scores['value_separation']:.4f}")
    
    # 6. Key Recommendations
    print("\n6. KEY RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    # Feature selection recommendations
    if len(feature_importance_df) > 0:
        top_3_features = feature_importance_df.head(3)['feature'].tolist()
        recommendations.append(f"Focus on top 3 features: {', '.join(top_3_features)}")
    
    # Model simplification
    if summary_df is not None and len(summary_df) > 1:
        models_with_small_drop = summary_df[summary_df['performance_drop'] <= 0.02]
        if len(models_with_small_drop) > 1:
            min_features = models_with_small_drop['features_kept'].min()
            recommendations.append(f"Model can be simplified to ~{min_features} features with minimal performance loss")
    
    # Tree interpretability
    if best_surrogate is not None and best_surrogate['test_fidelity'] > 0.7:
        recommendations.append(f"Use {best_surrogate['tree_depth']}-depth decision tree for interpretable approximation")
    
    # Feature categories
    loss_features = [f for f in features_labels if 'loss' in f.lower()]
    grad_features = [f for f in features_labels if 'grad' in f.lower()]
    if len(loss_features) > 0:
        recommendations.append("Loss-based features appear important for discrimination")
    if len(grad_features) > 0:
        recommendations.append("Gradient-based features provide complementary information")
    
    # Interaction insights
    if sorted_interactions is not None and len(sorted_interactions) > 0:
        top_interaction = sorted_interactions[0][1]['features']
        recommendations.append(f"Consider feature interaction: {top_interaction[0]} × {top_interaction[1]}")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # 7. Implementation Guidance
    print("\n7. IMPLEMENTATION GUIDANCE")
    print("-" * 40)
    print("For practical deployment:")
    
    # Simple model option
    if best_surrogate is not None and best_surrogate['test_fidelity'] > 0.7:
        print(f"  • Use decision tree (depth {best_surrogate['tree_depth']}) for interpretable predictions")
    
    # Feature-reduced RF option
    if summary_df is not None and len(summary_df) > 1:
        good_reduced = summary_df[summary_df['performance_drop'] <= 0.01]
        if len(good_reduced) > 1:
            best_reduced = good_reduced.loc[good_reduced['features_kept'].idxmin()]
            print(f"  • Use RF with {best_reduced['features_kept']} features for simplified high-performance model")
    
    # Full model for maximum performance
    print(f"  • Use full RF ({len(features_labels)} features) for maximum accuracy: {test_accuracy:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return {
        'baseline_performance': {'cv_mean': cv_scores.mean(), 'test_accuracy': test_accuracy},
        'top_features': feature_importance_df.head(5)['feature'].tolist(),
        'importance_correlations': {'rf_perm': rf_perm_corr, 'rf_tree': rf_tree_corr},
        'best_surrogate_performance': best_surrogate,
        'recommendations': recommendations
    }


def save_all_results(results_package, filename='rf_interpretability_results.pkl'):
    """Save all results for future reference"""
    with open(filename, 'wb') as f:
        pickle.dump(results_package, f)
    print(f"All results saved to: {filename}")