import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy.stats import wasserstein_distance, ks_2samp, entropy, skew, kurtosis
from scipy.spatial.distance import jensenshannon, pdist, squareform
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sentence_transformers import SentenceTransformer
import pickle
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def read_json(fpath: str):
    with open(fpath, 'r') as f:
        return json.load(f)

def compute_auc_dict(features_dict):
    """
    Compute AUCs for all feature pairs between splits in features_dict.
    Returns a dict: {f"{split0}_{split1}_{feat}": auc_value}
    """
    auc_dict = {}
    split_names = list(features_dict.keys())
    feature_names = features_dict[split_names[0]]['features_names']
    for i, split0 in enumerate(split_names):
        for j, split1 in enumerate(split_names):
            if split0 == split1:
                continue
            for feat_idx, feat in enumerate(feature_names):
                vals0 = features_dict[split0]['unnormalized_features_tensor'][:, feat_idx].numpy()
                vals1 = features_dict[split1]['unnormalized_features_tensor'][:, feat_idx].numpy()
                vals = np.concatenate([vals0, vals1])
                labels = np.concatenate([np.zeros(len(vals0)), np.ones(len(vals1))])
                try:
                    fpr, tpr, _ = roc_curve(labels, vals)
                    auc_score = auc(fpr, tpr)
                except Exception:
                    auc_score = 0.5
                auc_dict[f"{split0}_{split1}_{feat}"] = auc_score
    return auc_dict

def plot_landscape_results_from_features(features_dict, plot_base_dir="loss_landscape_plots"):
    """
    Plots histograms, ROC curves, and AUC heatmaps from features_dict using Plotly (interactive).
    Adds KDE line to each histogram.
    """
    import numpy as np
    from sklearn.metrics import roc_curve
    import os
    os.makedirs(plot_base_dir, exist_ok=True)
    split_names = list(features_dict.keys())
    feature_names = features_dict[split_names[0]]['features_names']
    auc_vals = compute_auc_dict(features_dict)
    plots = {}

    # Assign colors for splits
    split_colors = {
        'retain': 'blue',
        'holdout': 'green',
        'forget': 'red'
    }
    # Fallback for unknown splits
    default_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    for idx, split in enumerate(split_names):
        if split not in split_colors:
            split_colors[split] = default_colors[idx % len(default_colors)]

    # 1. Interactive Histograms for each feature and split, with KDE line
    for feat_idx, feat in enumerate(feature_names):
        fig = go.Figure()
        for split in split_names:
            color = split_colors.get(split, None)
            values = features_dict[split]['unnormalized_features_tensor'][:, feat_idx].numpy()
            fig.add_trace(go.Histogram(
                x=values,
                name=f"{split} histogram",
                opacity=0.6,
                histnorm='probability density',
                marker_color=color
            ))
            # Add KDE line only if variance is nonzero
            if np.std(values) > 0:
                try:
                    kde = gaussian_kde(values)
                    x_grid = np.linspace(values.min(), values.max(), 200)
                    y_grid = kde(x_grid)
                    fig.add_trace(go.Scatter(
                        x=x_grid,
                        y=y_grid,
                        mode='lines',
                        name=f"{split} KDE",
                        line=dict(width=3, color=color)
                    ))
                except Exception as e:
                    print(f"KDE error for {split} {feat}: {e}")
            else:
                print(f"KDE skipped for {split} {feat}: data is constant.")

        fig.update_layout(
            title=f"Histogram: {feat}",
            barmode='overlay',
            width=900,
            height=600
        )
        plots[f"hist_{feat.replace('/', '_')}"] = fig
        
    # 2. Interactive ROC curves and collect AUCs for heatmap
    auc_matrix = {feat: np.zeros((len(split_names), len(split_names))) for feat in feature_names}
    for i, split0 in enumerate(split_names):
        for j, split1 in enumerate(split_names):
            if split0 == split1:
                continue
            for feat_idx, feat in enumerate(feature_names):
                auc_key = f"{split0}_{split1}_{feat}"
                if auc_key in auc_vals:
                    auc_score = auc_vals[auc_key]
                    auc_matrix[feat][i, j] = auc_score
                    # ROC curve
                    vals0 = features_dict[split0]['unnormalized_features_tensor'][:, feat_idx].numpy()
                    vals1 = features_dict[split1]['unnormalized_features_tensor'][:, feat_idx].numpy()
                    vals = np.concatenate([vals0, vals1])
                    y = np.concatenate([np.zeros(len(vals0)), np.ones(len(vals1))])
                    try:
                        fpr, tpr, _ = roc_curve(y, vals)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {auc_score:.3f}"))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random", line=dict(dash='dash')))
                        fig.update_layout(
                            title=f"ROC Curve: {split0} vs {split1} - {feat}",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            width=900,
                            height=600
                        )
                        plots[f"roc_{auc_key.replace('/', '_')}"] = fig
                    except Exception as e:
                        print(f"Error creating ROC curve for {auc_key}: {e}")

    # 3. Interactive AUC Heatmaps for each feature
    for feat in feature_names:
        try:
            fig = px.imshow(
                auc_matrix[feat],
                x=split_names,
                y=split_names,
                color_continuous_scale="viridis",
                zmin=0.5,
                zmax=1.0,
                text_auto=".3f",
                aspect="auto"
            )
            fig.update_layout(
                title=f"AUC Heatmap: {feat}",
                width=900,
                height=700
            )
            plots[f"auc_heatmap_{feat.replace('/', '_')}"] = fig
        except Exception as e:
            print(f"Error creating heatmap for {feat}: {e}")

    return plots

def show_plots(plots: dict, plots_types=['hist', 'roc', 'auc_heatmap']):
    """
    Display plots from a dictionary. Handles image file paths and Plotly figures.
    """
    for plt_name, plt_obj in plots.items():
        if not any(pt.lower() in plt_name.lower() for pt in plots_types):
            continue

        # If nested dict, recurse
        if isinstance(plt_obj, dict):
            show_plots(plt_obj, plots_types)
        # If file path to image
        elif isinstance(plt_obj, str) and os.path.isfile(plt_obj):
            img = mpimg.imread(plt_obj)
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(plt_obj))
            plt.show()
        # If Plotly figure
        elif hasattr(plt_obj, 'show'):
            plt_obj.show()
    
def normalize_features(t1, t2=None, t3=None):
    """
    Normalize one or multiple feature tensors using the overall mean and standard deviation
    across all tensors. Each feature is normalized to have zero mean and unit variance.
    """
    tensors = [t for t in [t1, t2, t3] if t is not None]
    if not tensors or any(t.numel() == 0 for t in tensors):
        print("Warning: One or more input tensors are empty. Returning originals unchanged.")
    combined = torch.cat(tensors, dim=0)
    
    mean = combined.mean(dim=0, keepdim=True)
    std = combined.std(dim=0, keepdim=True)
    std[std == 0] = 1

    def normalize(t):
        return (t - mean) / std

    normalized_t1 = normalize(t1)
    normalized_t2 = normalize(t2) if t2 is not None else None
    normalized_t3 = normalize(t3) if t3 is not None else None

    return normalized_t1, normalized_t2, normalized_t3

def train_predictors(retain_t, holdout_t, features_labels, forget_t=None, test_size=0.15):
    """
    Train membership inference classifiers for the provided tensors.
    """
    if retain_t.numel() == 0 or holdout_t.numel() == 0 or (forget_t is not None and forget_t.numel() == 0):
        print("Warning: One or more tensors are empty. Skipping training.")
    results = {}
    feature_importance_results = {}
    trained_classifiers = {}

    datasets = {'retain': retain_t, 'holdout': holdout_t}
    if forget_t is not None:
        datasets['forget'] = forget_t

    tensors = [tensor for tensor in datasets.values()]
    labels = [torch.full((len(tensor),), i) for i, tensor in enumerate(datasets.values())]

    X = torch.cat(tensors, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Logistic Regression
    clf_logistic = LogisticRegression(random_state=42, max_iter=300, multi_class='multinomial', solver='lbfgs')
    clf_logistic.fit(X_train, y_train)
    y_pred_logistic = clf_logistic.predict(X_test)
    results['multi_class'] = {'logistic': {
        'accuracy': accuracy_score(y_test, y_pred_logistic),
        'f1': f1_score(y_test, y_pred_logistic, average='weighted'),
        'roc_auc': roc_auc_score(y_test, clf_logistic.predict_proba(X_test), multi_class='ovr'),
        'model': clf_logistic
    }}
    importance_logistic = abs(clf_logistic.coef_).mean(axis=0)
    feature_importance_results['multi_class_logistic'] = pd.DataFrame({
        'feature': features_labels,
        'importance': importance_logistic / np.sum(importance_logistic)
    }).sort_values('importance', ascending=False)

    # Random Forest
    clf_rf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    results['multi_class']['random_forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf, average='weighted'),
        'roc_auc': roc_auc_score(y_test, clf_rf.predict_proba(X_test), multi_class='ovr'),
        'model': clf_rf
    }
    importance_rf = clf_rf.feature_importances_
    feature_importance_results['multi_class_rf'] = pd.DataFrame({
        'feature': features_labels,
        'importance': importance_rf / np.sum(importance_rf)
    }).sort_values('importance', ascending=False)
    
    trained_classifiers['logistic'] = clf_logistic
    trained_classifiers['random_forest'] = clf_rf

    return results, feature_importance_results, trained_classifiers

def plot_classification_results(results, feature_importance_results, save_dir="classification_plots"):
    os.makedirs(save_dir, exist_ok=True)
    plots = {}
    
    performance_data = []
    for clf in ['logistic', 'random_forest']:
        for metric in ['accuracy', 'f1', 'roc_auc']:
            performance_data.append({
                'Classifier': clf.replace('_', ' ').title(),
                'Metric': metric.upper(),
                'Score': results['multi_class'][clf][metric]
            })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(performance_data), x='Metric', y='Score', hue='Classifier')
    plt.title('Multi-Class Classification Performance')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    performance_path = os.path.join(save_dir, 'performance_comparison.png')
    plt.savefig(performance_path, bbox_inches='tight', dpi=150)
    plt.show()
    plots['performance_comparison'] = performance_path
    
    for clf_key, importance_df in feature_importance_results.items():
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', orient='h')
        plt.title(f'Top 15 Feature Importance - {clf_key.replace("_", " ").title()}')
        importance_path = os.path.join(save_dir, f'feature_importance_{clf_key}.png')
        plt.savefig(importance_path, bbox_inches='tight', dpi=150)
        plt.show()
        plots[f'feature_importance_{clf_key}'] = importance_path
        
    return plots

def plot_confusion_matrices(results, norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, save_dir="classification_plots", test_size=0.15):
    datasets = {'retain': norm_retain_tensor, 'holdout': norm_holdout_tensor, 'forget': norm_forget_tensor}
    tensors = [t for t in datasets.values()]
    labels = [torch.full((len(t),), i) for i, t in enumerate(datasets.values())]
    class_names = list(datasets.keys())
    X = torch.cat(tensors, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    for clf_name in ['logistic', 'random_forest']:
        model = results['multi_class'][clf_name]['model']
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {clf_name.replace("_", " ").title()}')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        cm_path = os.path.join(save_dir, f'confusion_matrix_{clf_name}.png')
        plt.savefig(cm_path, bbox_inches='tight', dpi=150)
        plt.show()
        
        print(f"\\n=== Classification Report - {clf_name.replace('_', ' ').title()} ===")
        print(classification_report(y_test, y_pred, target_names=class_names))

def analyze_prediction_confidence(results, norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, save_dir="classification_plots", test_size=0.15):
    datasets = {'retain': norm_retain_tensor, 'holdout': norm_holdout_tensor, 'forget': norm_forget_tensor}
    tensors = [t for t in datasets.values()]
    labels = [torch.full((len(t),), i) for i, t in enumerate(datasets.values())]
    class_names = list(datasets.keys())
    X = torch.cat(tensors, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    for clf_name in ['logistic', 'random_forest']:
        model = results['multi_class'][clf_name]['model']
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        confidence_df = pd.DataFrame([{
            'True_Class': class_names[true_label], 'Predicted_Class': class_names[pred_label],
            'Confidence': np.max(probas), 'Correct': true_label == pred_label
        } for true_label, pred_label, probas in zip(y_test, y_pred, y_proba)])
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(data=confidence_df, x='Confidence', hue='Correct', bins=20, alpha=0.7)
        plt.title(f'Prediction Confidence Distribution - {clf_name.replace("_", " ").title()}')
        plt.subplot(1, 2, 2)
        sns.boxplot(data=confidence_df, x='True_Class', y='Confidence', hue='Correct')
        plt.title(f'Confidence by True Class - {clf_name.replace("_", " ").title()}')
        confidence_path = os.path.join(save_dir, f'confidence_analysis_{clf_name}.png')
        plt.savefig(confidence_path, bbox_inches='tight', dpi=150)
        plt.show()

def plot_multiclass_roc_curves(results, norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, save_dir="classification_plots", test_size=0.15):
    datasets = {'retain': norm_retain_tensor, 'holdout': norm_holdout_tensor, 'forget': norm_forget_tensor}
    tensors = [t for t in datasets.values()]
    labels = [torch.full((len(t),), i) for i, t in enumerate(datasets.values())]
    class_names = list(datasets.keys())
    X = torch.cat(tensors, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()
    
    y_bin = label_binarize(y, classes=list(range(len(class_names))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    
    for clf_name in ['logistic', 'random_forest']:
        model = results['multi_class'][clf_name]['model']
        y_score = model.predict_proba(X_test)
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        clf.predict_proba(X_test)[:, 1]
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC of class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.title(f'Multi-class ROC Curves - {clf_name.replace("_", " ").title()}')
        plt.legend(loc="lower right")
        roc_path = os.path.join(save_dir, f'multiclass_roc_{clf_name}.png')
        plt.savefig(roc_path, bbox_inches='tight', dpi=150)
        plt.show()

def train_binary_comparisons(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, features_labels, save_dir="classification_plots", test_size=0.15):
    if norm_retain_tensor.numel() == 0 or norm_holdout_tensor.numel() == 0 or (norm_forget_tensor is not None and norm_forget_tensor.numel() == 0):
        print("Warning: One or more tensors are empty. Skipping training.")
    binary_results = {}
    binary_feature_importance = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    comparisons = {
        'holdout_vs_all': {'positive': norm_holdout_tensor, 'negative': torch.cat([norm_retain_tensor, norm_forget_tensor])},
        'forget_vs_all': {'positive': norm_forget_tensor, 'negative': torch.cat([norm_retain_tensor, norm_holdout_tensor])}
    }
    
    trained_classifiers = {}
    for name, data in comparisons.items():
        X = torch.cat([data['positive'], data['negative']]).numpy()
        y = torch.cat([torch.ones(len(data['positive'])), torch.zeros(len(data['negative']))]).numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        binary_results[name] = {}
        tmp_trained_classifiers = {}
        for clf_type, (clf_class, params) in [('logistic', (LogisticRegression, {'random_state': 42, 'max_iter': 300})), ('random_forest', (RandomForestClassifier, {'random_state': 42, 'n_estimators': 100}))]:
            clf = clf_class(**params).fit(X_train, y_train)
            y_pred, y_proba = clf.predict(X_test), clf.predict_proba(X_test)[:, 1]
            
            binary_results[name][clf_type] = {'accuracy': accuracy_score(y_test, y_pred), 'f1': f1_score(y_test, y_pred), 'roc_auc': roc_auc_score(y_test, y_proba), 'model': clf}
            
            importance = abs(clf.coef_[0]) if clf_type == 'logistic' else clf.feature_importances_
            binary_feature_importance[f"{name}_{clf_type}"] = pd.DataFrame({'feature': features_labels, 'importance': importance / np.sum(importance)}).sort_values('importance', ascending=False)
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {binary_results[name][clf_type]["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'ROC Curve: {name.replace("_", " ").title()} - {clf_type.replace("_", " ").title()}')
            plt.legend(loc="lower right")
            roc_path = os.path.join(save_dir, f'binary_roc_{name}_{clf_type}.png')
            plt.savefig(roc_path, bbox_inches='tight', dpi=150)
            plt.show()

            tmp_trained_classifiers[clf_type] = clf
    
        trained_classifiers[name] = tmp_trained_classifiers

    return binary_results, binary_feature_importance, trained_classifiers

def create_comprehensive_summary(results, binary_results, feature_importance_results, binary_feature_importance, save_dir="classification_plots"):
    performance_data = []
    for clf_type in ['logistic', 'random_forest']:
        metrics = results['multi_class'][clf_type]
        for metric_name in ['accuracy', 'f1', 'roc_auc']:
            performance_data.append({'Classifier': clf_type.title(), 'Task': 'Multi-Class', 'Metric': metric_name.upper(), 'Score': metrics[metric_name]})
    
    for comparison in ['holdout_vs_all', 'forget_vs_all']:
        for clf_type in ['logistic', 'random_forest']:
            metrics = binary_results[comparison][clf_type]
            for metric_name in ['accuracy', 'f1', 'roc_auc']:
                performance_data.append({'Classifier': clf_type.title(), 'Task': comparison.replace('_', ' ').title(), 'Metric': metric_name.upper(), 'Score': metrics[metric_name]})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, metric in enumerate(['ACCURACY', 'F1', 'ROC_AUC']):
        sns.barplot(data=pd.DataFrame(performance_data)[lambda df: df['Metric'] == metric], x='Task', y='Score', hue='Classifier', ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'comprehensive_performance_summary.png')
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.show()
    plots = {'performance_comparison': summary_path}
    
    return plots

def compute_statistical_distances(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, features_labels):
    """
    Compute various statistical distance measures between feature distributions.
    """
    distances = {}
    
    datasets = {
        'retain': norm_retain_tensor,
        'holdout': norm_holdout_tensor, 
        'forget': norm_forget_tensor
    }
    
    for i, (name1, tensor1) in enumerate(datasets.items()):
        for j, (name2, tensor2) in enumerate(datasets.items()):
            if i >= j:
                continue
                
            pair_distances = {}
            
            for feat_idx, feat_name in enumerate(features_labels):
                vals1 = tensor1[:, feat_idx].numpy()
                vals2 = tensor2[:, feat_idx].numpy()
                
                # Wasserstein distance (Earth Mover's Distance)
                wasserstein_dist = wasserstein_distance(vals1, vals2)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = ks_2samp(vals1, vals2)
                
                # Jensen-Shannon divergence (requires binning)
                hist1, bins = np.histogram(vals1, bins=20, density=True)
                hist2, _ = np.histogram(vals2, bins=bins, density=True)
                hist1 = hist1 + 1e-10  # Avoid zeros
                hist2 = hist2 + 1e-10
                js_div = jensenshannon(hist1, hist2)
                
                pair_distances[feat_name] = {
                    'wasserstein': wasserstein_dist,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'jensen_shannon': js_div
                }
            
            distances[f"{name1}_vs_{name2}"] = pair_distances
    
    return distances

def plot_statistical_distances(stat_distances, features_labels, plots_base_dir="loss_landscape_plots"):
    """Create heatmaps for different statistical distance measures."""
    
    # Prepare data for each metric
    metrics = ['wasserstein', 'ks_statistic', 'jensen_shannon']
    metric_names = ['Wasserstein Distance', 'KS Statistic', 'Jensen-Shannon Divergence']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        # Create matrix for this metric
        data_matrix = []
        
        for feat_name in features_labels:
            row = []
            for comparison in stat_distances.keys():
                row.append(stat_distances[comparison][feat_name][metric])
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        im = axes[idx].imshow(data_matrix, cmap='viridis', aspect='auto')
        axes[idx].set_title(f'{metric_name}\\nAcross Feature Types', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Dataset Comparisons')
        axes[idx].set_ylabel('Features')
        
        # Set labels
        axes[idx].set_xticks(range(len(stat_distances.keys())))
        axes[idx].set_xticklabels(list(stat_distances.keys()), rotation=45, ha='right')
        axes[idx].set_yticks(range(len(features_labels)))
        axes[idx].set_yticklabels(features_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
        
        # Add text annotations
        for i in range(len(features_labels)):
            for j in range(len(stat_distances.keys())):
                text = axes[idx].text(j, i, f'{data_matrix[i, j]:.3f}',
                                    ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    os.makedirs(plots_base_dir, exist_ok=True)
    fig_path = os.path.join(plots_base_dir, 'statistical_distances_heatmap.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.show()

def compare_manifold_structures(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, plots_base_dir="loss_landscape_plots"):
    """
    Compare the manifold structure of different datasets using dimensionality reduction.
    """
    # Combine all data
    all_data = torch.cat([norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor], dim=0).numpy()
    labels = (['retain'] * len(norm_retain_tensor) + 
              ['holdout'] * len(norm_holdout_tensor) + 
              ['forget'] * len(norm_forget_tensor))
    
    manifold_results = {}
    
    # Different manifold learning techniques
    techniques = {
        'PCA': PCA(n_components=2),
        'TSNE': TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data)//4)),
        'Isomap': Isomap(n_components=2)
    }
    
    # Add UMAP if available
    if UMAP_AVAILABLE:
        techniques['UMAP'] = umap.UMAP(n_components=2, random_state=42)
    
    n_techniques = len(techniques)
    cols = 2
    rows = (n_techniques + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    colors = {'retain': 'blue', 'holdout': 'green', 'forget': 'red'}
    
    for idx, (name, technique) in enumerate(techniques.items()):
        print(f"Computing {name} embedding...")
        embedding = technique.fit_transform(all_data)
        
        # Plot
        ax = axes[idx]
        for label in ['retain', 'holdout', 'forget']:
            mask = np.array(labels) == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                    label=label, alpha=0.6, s=30, c=colors[label])
        
        ax.set_title(f'{name} Embedding\\nDataset Separation in 2D Space', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f'{name} Component 1')
        ax.set_ylabel(f'{name} Component 2')
        
        manifold_results[name] = {
            'embedding': embedding,
            'labels': labels
        }
    
    # Hide unused subplots
    for idx in range(len(techniques), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{plots_base_dir}/manifold_comparisons.png', bbox_inches='tight', dpi=150)
    plt.show()
    
    return manifold_results

def analyze_clustering_patterns(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, plots_base_dir="loss_landscape_plots"):
    """
    Analyze clustering patterns and separability between datasets.
    """
    datasets = {
        'retain': norm_retain_tensor,
        'holdout': norm_holdout_tensor,
        'forget': norm_forget_tensor
    }
    
    clustering_results = {}
    
    # Test different numbers of clusters
    n_clusters_range = range(2, min(8, min(len(tensor) for tensor in datasets.values()) // 2))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = {'retain': 'blue', 'holdout': 'green', 'forget': 'red'}
    
    for name, tensor in datasets.items():
        data = tensor.numpy()
        
        silhouette_scores = []
        inertias = []
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        clustering_results[name] = {
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'n_clusters_range': list(n_clusters_range)
        }
        
        # Plot results
        ax1.plot(n_clusters_range, silhouette_scores, 
                marker='o', label=f'{name} Silhouette', color=colors[name], linewidth=2)
        ax2.plot(n_clusters_range, inertias, 
                marker='s', label=f'{name} Inertia', color=colors[name], linewidth=2)
    
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Analysis\\nHigher = Better Defined Clusters', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Inertia')
    ax2.set_title('Elbow Method\\nLower = More Compact Clusters', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plots_base_dir}/clustering_analysis.png', bbox_inches='tight', dpi=150)
    plt.show()
    
    return clustering_results

def analyze_landscape_topology(features_dict):
    """
    Analyze topological properties of the loss landscape.
    """
    topology_results = {}
    
    for split_name, split_data in features_dict.items():
        tensor = split_data['unnormalized_features_tensor']
        feature_names = split_data['features_names']
        
        split_topology = {}
        
        for feat_idx, feat_name in enumerate(feature_names):
            values = tensor[:, feat_idx].numpy()
            
            # Basic statistical moments
            split_topology[feat_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'skewness': skew(values),
                'kurtosis': kurtosis(values),
                'range': np.ptp(values),  # Peak-to-peak
                'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            }
        
        # Pairwise feature correlations
        correlation_matrix = np.corrcoef(tensor.numpy().T)
        split_topology['feature_correlations'] = correlation_matrix
        
        # Distance-based metrics
        distances = pdist(tensor.numpy())
        split_topology['distance_stats'] = {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances)
        }
        
        topology_results[split_name] = split_topology
    
    return topology_results

def plot_topology_comparison(topology_results, features_labels, plots_base_dir="loss_landscape_plots"):
    """Plot comparison of topological properties across datasets."""
    
    # Extract statistical moments for comparison
    properties = ['mean', 'std', 'skewness', 'kurtosis', 'range', 'iqr', 'cv']
    datasets = list(topology_results.keys())
    
    # Create subplots for each property
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    colors = {'retain': 'blue', 'holdout': 'green', 'forget': 'red'}
    
    for prop_idx, prop in enumerate(properties):
        ax = axes[prop_idx]
        
        # Collect data for this property
        x_positions = np.arange(len(features_labels))
        width = 0.25
        
        for dataset_idx, dataset in enumerate(datasets):
            values = []
            for feat_name in features_labels:
                if feat_name in topology_results[dataset]:
                    values.append(topology_results[dataset][feat_name][prop])
                else:
                    values.append(0)
            
            offset = (dataset_idx - 1) * width
            bars = ax.bar(x_positions + offset, values, width, 
                         label=dataset, color=colors.get(dataset, 'gray'), alpha=0.7)
        
        ax.set_title(f'{prop.title()} Comparison', fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel(f'{prop.title()} Value')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(features_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot if odd number of properties
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{plots_base_dir}/topology_comparison.png', bbox_inches='tight', dpi=150)
    plt.show()

def analyze_landscape_smoothness(features_dict):
    """
    Analyze the smoothness/roughness of the loss landscape.
    """
    smoothness_results = {}
    
    for split_name, split_data in features_dict.items():
        tensor = split_data['unnormalized_features_tensor']
        feature_names = split_data['features_names']
        
        smoothness_metrics = {}
        
        for feat_idx, feat_name in enumerate(feature_names):
            values = tensor[:, feat_idx].numpy()
            
            # Sort values to analyze smoothness
            sorted_values = np.sort(values)
            
            # First and second differences
            first_diff = np.diff(sorted_values)
            second_diff = np.diff(first_diff) if len(first_diff) > 1 else np.array([0])
            
            # Smoothness metrics
            smoothness_metrics[feat_name] = {
                'first_diff_var': np.var(first_diff) if len(first_diff) > 0 else 0,
                'second_diff_var': np.var(second_diff) if len(second_diff) > 0 else 0,
                'gradient_magnitude': np.mean(np.abs(first_diff)) if len(first_diff) > 0 else 0,
                'curvature_estimate': np.mean(np.abs(second_diff)) if len(second_diff) > 0 else 0,
                'roughness_index': np.sum(np.abs(second_diff)) / len(second_diff) if len(second_diff) > 0 else 0
            }
        
        smoothness_results[split_name] = smoothness_metrics
    
    return smoothness_results

def plot_smoothness_comparison(smoothness_results, features_labels, plots_base_dir="loss_landscape_plots"):
    """Plot comparison of smoothness properties across datasets."""
    
    smoothness_metrics = ['first_diff_var', 'second_diff_var', 'gradient_magnitude', 'curvature_estimate', 'roughness_index']
    metric_names = ['First Diff Variance', 'Second Diff Variance', 'Gradient Magnitude', 'Curvature Estimate', 'Roughness Index']
    datasets = list(smoothness_results.keys())
    
    # Create subplots for each smoothness metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = {'retain': 'blue', 'holdout': 'green', 'forget': 'red'}
    
    for metric_idx, (metric, metric_name) in enumerate(zip(smoothness_metrics, metric_names)):
        ax = axes[metric_idx]
        
        # Collect data for this metric
        x_positions = np.arange(len(features_labels))
        width = 0.25
        
        for dataset_idx, dataset in enumerate(datasets):
            values = []
            for feat_name in features_labels:
                if feat_name in smoothness_results[dataset]:
                    values.append(smoothness_results[dataset][feat_name][metric])
                else:
                    values.append(0)
            
            offset = (dataset_idx - 1) * width
            bars = ax.bar(x_positions + offset, values, width, 
                         label=dataset, color=colors.get(dataset, 'gray'), alpha=0.7)
        
        ax.set_title(f'{metric_name}\\nLower = Smoother Landscape', fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel(f'{metric_name}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(features_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{plots_base_dir}/smoothness_comparison.png', bbox_inches='tight', dpi=150)
    plt.show()

def analyze_prediction_transfer(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor):
    """
    Analyze how well models trained on one dataset generalize to others.
    """
    datasets = {
        'retain': norm_retain_tensor,
        'holdout': norm_holdout_tensor,
        'forget': norm_forget_tensor
    }
    
    transfer_results = {}
    
    # For each pair of datasets
    for train_name, train_tensor in datasets.items():
        for test_name, test_tensor in datasets.items():
            if train_name == test_name:
                continue
            
            # Create binary classification problem (train_set vs others)
            other_datasets = [v for k, v in datasets.items() if k != train_name]
            negative_data = torch.cat(other_datasets, dim=0)
            
            # Train data
            X_train = torch.cat([train_tensor, negative_data], dim=0).numpy()
            y_train = np.concatenate([np.ones(len(train_tensor)), np.zeros(len(negative_data))])
            
            # Test data (binary: test_set vs others)
            other_test = [v for k, v in datasets.items() if k != test_name]
            negative_test = torch.cat(other_test, dim=0)
            
            X_test = torch.cat([test_tensor, negative_test], dim=0).numpy()
            y_test = np.concatenate([np.ones(len(test_tensor)), np.zeros(len(negative_test))])
            
            # Train model
            clf = LogisticRegression(random_state=42, max_iter=300)
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_proba)
            
            transfer_results[f"train_{train_name}_test_{test_name}"] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'model': clf
            }
    
    return transfer_results

def plot_transfer_results(transfer_results, plots_base_dir="loss_landscape_plots"):
    """Plot transfer learning results as heatmaps."""
    
    datasets = ['retain', 'holdout', 'forget']
    
    # Create matrices for accuracy and AUC
    accuracy_matrix = np.zeros((3, 3))
    auc_matrix = np.zeros((3, 3))
    
    for i, train_dataset in enumerate(datasets):
        for j, test_dataset in enumerate(datasets):
            if train_dataset == test_dataset:
                accuracy_matrix[i, j] = np.nan  # Same dataset
                auc_matrix[i, j] = np.nan
            else:
                key = f"train_{train_dataset}_test_{test_dataset}"
                if key in transfer_results:
                    accuracy_matrix[i, j] = transfer_results[key]['accuracy']
                    auc_matrix[i, j] = transfer_results[key]['auc']
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy heatmap
    im1 = ax1.imshow(accuracy_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax1.set_title('Cross-Dataset Transfer Accuracy\\nHigher = Better Transfer', fontweight='bold')
    ax1.set_xlabel('Test Dataset')
    ax1.set_ylabel('Train Dataset')
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(datasets)
    ax1.set_yticklabels(datasets)
    
    # Add text annotations for accuracy
    for i in range(3):
        for j in range(3):
            if not np.isnan(accuracy_matrix[i, j]):
                text = ax1.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # AUC heatmap
    im2 = ax2.imshow(auc_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax2.set_title('Cross-Dataset Transfer AUC\\nHigher = Better Discrimination', fontweight='bold')
    ax2.set_xlabel('Test Dataset')
    ax2.set_ylabel('Train Dataset')
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(datasets)
    ax2.set_yticklabels(datasets)
    
    # Add text annotations for AUC
    for i in range(3):
        for j in range(3):
            if not np.isnan(auc_matrix[i, j]):
                text = ax2.text(j, i, f'{auc_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{plots_base_dir}/transfer_learning_results.png', bbox_inches='tight', dpi=150)
    plt.show()

def create_landscape_summary_report(stat_distances, topology_results, smoothness_results, transfer_results, clustering_results):
    """Create a comprehensive summary report of all landscape comparisons."""
    print("="*80)
    print("COMPREHENSIVE LANDSCAPE COMPARISON SUMMARY")
    print("="*80)
    
    # Statistical Distances Summary
    print("\\n1. STATISTICAL DISTANCES SUMMARY:")
    print("-" * 40)
    for comparison, distances in stat_distances.items():
        print(f"\\n{comparison.upper()}:")
        
        # Average across all features for each metric
        wasserstein_avg = np.mean([d['wasserstein'] for d in distances.values()])
        ks_avg = np.mean([d['ks_statistic'] for d in distances.values()])
        js_avg = np.mean([d['jensen_shannon'] for d in distances.values()])
        
        print(f"  Average Wasserstein Distance: {wasserstein_avg:.4f}")
        print(f"  Average KS Statistic: {ks_avg:.4f}")
        print(f"  Average Jensen-Shannon Divergence: {js_avg:.4f}")
    
    # Topology Summary
    print("\\n2. TOPOLOGICAL PROPERTIES SUMMARY:")
    print("-" * 40)
    for dataset, topology in topology_results.items():
        print(f"\\n{dataset.upper()}:")
        for prop, value in topology.items():
            if prop == 'feature_correlations':
                continue  # Skip correlations for now
            if isinstance(value, (float, int, np.floating, np.integer)):
                print(f"  {prop.title()}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {prop.title()}:")
                for k, v in value.items():
                    if isinstance(v, (float, int, np.floating, np.integer)):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {prop.title()}: {value}")
        
        # Feature correlations
        corr = topology['feature_correlations']
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title(f"Feature Correlations - {dataset.title()}")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    # Silhouette and Inertia Summary
    print("\\n3. CLUSTERING PATTERNS SUMMARY:")
    print("-" * 40)
    for dataset, results in clustering_results.items():
        best_silhouette = max(results['silhouette_scores'])
        best_inertia = min(results['inertias'])
        print(f"  {dataset.title()}: Best Silhouette = {best_silhouette:.4f}, Best Inertia = {best_inertia:.4f}")
    
    # Smoothness Summary
    print("\\n4. SMOOTHNESS ANALYSIS SUMMARY:")
    print("-" * 40)
    for dataset, features_metrics in smoothness_results.items():
        print(f"  {dataset.title()}:")
        # Calculate average metrics across all features
        all_metrics = {}
        for feature_name, metrics in features_metrics.items():
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Print average metrics
        for metric_name, values in all_metrics.items():
            avg_value = np.mean(values)
            print(f"    {metric_name.replace('_', ' ').title()}: {avg_value:.4f}")
    
    # Transfer Results Summary
    print("\\n5. TRANSFER LEARNING RESULTS SUMMARY:")
    print("-" * 40)
    for key, results in transfer_results.items():
        print(f"  {key}: Accuracy = {results['accuracy']:.4f}, AUC = {results['auc']:.4f}")
    
    print("\n" + "="*80)

def analyze_ill_feature_discrimination(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, features_labels):
    """
    Analyze which ILL features are most discriminative between forget/retain/holdout.
    """
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    
    # Combine all data
    X = torch.cat([norm_forget_tensor, norm_retain_tensor, norm_holdout_tensor], dim=0).numpy()
    y = (['forget'] * len(norm_forget_tensor) + 
         ['retain'] * len(norm_retain_tensor) + 
         ['holdout'] * len(norm_holdout_tensor))
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Feature importance analysis
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_encoded)
    
    # Cross-validation scores
    cv_scores = cross_val_score(rf, X, y_encoded, cv=5, scoring='accuracy')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features_labels,
        'importance': rf.feature_importances_,
        'rank': range(1, len(features_labels) + 1)
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    feature_importance['rank'] = range(1, len(feature_importance) + 1)
    
    # Predictions for confusion matrix
    y_pred = rf.predict(X)
    
    print("ILL FEATURE DISCRIMINATION ANALYSIS")
    print("="*50)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"Overall classification accuracy: {rf.score(X, y_encoded):.3f}")
    print("\nTop 5 Most Discriminative ILL Features:")
    for i, row in feature_importance.head().iterrows():
        print(f"{row['rank']}. {row['feature']}: {row['importance']:.4f}")
    
    return {
        'feature_importance': feature_importance,
        'cv_scores': cv_scores,
        'model': rf,
        'predictions': y_pred,
        'true_labels': y_encoded,
        'label_encoder': le
    }

def plot_ill_discrimination_analysis(ill_discrimination, features_labels, plots_base_dir="loss_landscape_plots"):
    """
    Visualize ILL feature discrimination results.
    """
    feature_importance = ill_discrimination['feature_importance']
    
    # Create comprehensive discrimination plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # 1. Feature Importance Bar Plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    bars = ax1.barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
    ax1.set_yticks(range(len(feature_importance)))
    ax1.set_yticklabels(feature_importance['feature'])
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('ILL Feature Importance for Dataset Discrimination\n(Random Forest)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # 2. Feature Importance by Category
    feature_categories = {
        'Loss-based': ['original_loss', 'mean_neighbor_loss', 'max_neighbor_loss', 'min_neighbor_loss'],
        'Variance-based': ['loss_variance', 'loss_std', 'increment_variance'],
        'Gradient-based': ['mean_gradient', 'max_gradient', 'gradient_variance'],
        'Landscape': ['loss_volatility', 'local_curvature'],
        'Increment-based': ['mean_loss_increment', 'max_loss_increment', 'min_loss_increment']
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_features = feature_importance[feature_importance['feature'].isin(features)]
        category_importance[category] = category_features['importance'].sum()
    
    categories = list(category_importance.keys())
    importances = list(category_importance.values())
    
    bars2 = ax2.bar(categories, importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax2.set_title('ILL Feature Category Importance\nfor Dataset Discrimination', fontweight='bold')
    ax2.set_ylabel('Cumulative Importance')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ill_discrimination['true_labels'], ill_discrimination['predictions'])
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax3.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax3.set_title('Dataset Classification Confusion Matrix\n(Normalized)', fontweight='bold')
    
    class_names = ['forget', 'retain', 'holdout']
    tick_marks = np.arange(len(class_names))
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(class_names)
    ax3.set_yticklabels(class_names)
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm_normalized.shape):
        ax3.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontweight='bold')
    
    plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # 4. Cross-validation scores
    cv_scores = ill_discrimination['cv_scores']
    ax4.boxplot([cv_scores], labels=['5-Fold CV'])
    ax4.scatter([1] * len(cv_scores), cv_scores, color='red', alpha=0.7, s=50)
    ax4.set_ylabel('Accuracy')
    ax4.set_title(f'Cross-Validation Performance\nMean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}', 
                 fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # Add mean line
    ax4.axhline(y=cv_scores.mean(), color='blue', linestyle='--', alpha=0.7, 
               label=f'Mean: {cv_scores.mean():.3f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{plots_base_dir}/ill_discrimination_analysis.png', bbox_inches='tight', dpi=150)
    plt.show()

def analyze_loss_landscape_topology_discrimination(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, features_labels):
    """
    Specialized analysis of loss landscape topology for discrimination.
    """
    datasets = {
        'forget': norm_forget_tensor,
        'retain': norm_retain_tensor,
        'holdout': norm_holdout_tensor
    }
    
    # Define feature groups for targeted analysis
    feature_groups = {
        'Loss Features': ['original_loss', 'mean_neighbor_loss', 'max_neighbor_loss', 'min_neighbor_loss'],
        'Variance Features': ['loss_variance', 'loss_std', 'increment_variance'],
        'Gradient Features': ['mean_gradient', 'max_gradient', 'gradient_variance'],
        'Landscape Features': ['loss_volatility', 'local_curvature'],
        'Increment Features': ['mean_loss_increment', 'max_loss_increment', 'min_loss_increment']
    }
    
    discrimination_results = {}
    
    print("LOSS LANDSCAPE TOPOLOGY DISCRIMINATION ANALYSIS")
    print("="*55)
    
    for group_name, group_features in feature_groups.items():
        print(f"\n{group_name.upper()}:")
        print("-" * 30)
        
        group_results = {}
        
        # Get indices for this feature group
        feature_indices = [i for i, feat in enumerate(features_labels) if feat in group_features]
        
        if not feature_indices:
            continue
            
        for dataset_name, tensor in datasets.items():
            group_data = tensor[:, feature_indices].numpy()
            
            # Compute discrimination metrics for this group
            group_results[dataset_name] = {
                'mean_values': np.mean(group_data, axis=0),
                'std_values': np.std(group_data, axis=0),
                'median_values': np.median(group_data, axis=0),
                'q75_values': np.percentile(group_data, 75, axis=0),
                'q25_values': np.percentile(group_data, 25, axis=0),
                'range_values': np.ptp(group_data, axis=0),
                'feature_names': [features_labels[i] for i in feature_indices]
            }
        
        # Calculate pairwise discrimination potential
        pairs = [('forget', 'retain'), ('forget', 'holdout'), ('retain', 'holdout')]
        
        for pair in pairs:
            data1 = datasets[pair[0]][:, feature_indices].numpy()
            data2 = datasets[pair[1]][:, feature_indices].numpy()
            
            # Effect size (Cohen's d) for each feature in the group
            effect_sizes = []
            for i in range(len(feature_indices)):
                pooled_std = np.sqrt((np.var(data1[:, i]) + np.var(data2[:, i])) / 2)
                if pooled_std > 0:
                    cohens_d = (np.mean(data1[:, i]) - np.mean(data2[:, i])) / pooled_std
                    effect_sizes.append(abs(cohens_d))
                else:
                    effect_sizes.append(0)
            
            avg_effect_size = np.mean(effect_sizes)
            max_effect_size = np.max(effect_sizes)
            
            print(f"  {pair[0]} vs {pair[1]}:")
            print(f"    Average effect size: {avg_effect_size:.3f}")
            print(f"    Max effect size: {max_effect_size:.3f}")
            
            # Store for later use
            group_results[f"{pair[0]}_vs_{pair[1]}_effect_sizes"] = effect_sizes
            group_results[f"{pair[0]}_vs_{pair[1]}_avg_effect"] = avg_effect_size
        
        discrimination_results[group_name] = group_results
    
    return discrimination_results

def analyze_loss_landscape_boundaries(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, features_labels):
    """
    Analyze decision boundaries in the loss landscape for dataset discrimination.
    """
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    
    # Combine data
    X = torch.cat([norm_forget_tensor, norm_retain_tensor, norm_holdout_tensor], dim=0).numpy()
    y = np.array(['forget'] * len(norm_forget_tensor) + 
                 ['retain'] * len(norm_retain_tensor) + 
                 ['holdout'] * len(norm_holdout_tensor))
    
    # Different classifiers to understand boundary complexity
    classifiers = {
        'Linear SVM': SVC(kernel='linear', random_state=42),
        'RBF SVM': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    boundary_results = {}
    
    print("LOSS LANDSCAPE BOUNDARY ANALYSIS")
    print("="*40)
    print("Testing different classifiers to understand decision boundary complexity...")
    
    for clf_name, clf in classifiers.items():
        # Cross-validation
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        
        # Fit and predict
        clf.fit(X, y)
        y_pred = clf.predict(X)
        train_accuracy = accuracy_score(y, y_pred)
        
        boundary_results[clf_name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'model': clf
        }
        
        print(f"{clf_name:15}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}, Train={train_accuracy:.3f}")
    
    return boundary_results

def analyze_pairwise_discrimination(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, features_labels):
    """
    Detailed pairwise analysis between forget/retain/holdout datasets.
    """
    from scipy.stats import ttest_ind, mannwhitneyu
    from sklearn.model_selection import cross_val_score
    
    datasets = {
        'forget': norm_forget_tensor,
        'retain': norm_retain_tensor,
        'holdout': norm_holdout_tensor
    }
    
    pairs = [('forget', 'retain'), ('forget', 'holdout'), ('retain', 'holdout')]
    pairwise_results = {}
    
    print("PAIRWISE DISCRIMINATION ANALYSIS")
    print("="*40)
    
    for pair in pairs:
        print(f"\n{pair[0].upper()} vs {pair[1].upper()}:")
        print("-" * 25)
        
        data1 = datasets[pair[0]].numpy()
        data2 = datasets[pair[1]].numpy()
        
        pair_results = {}
        
        # Combine for binary classification
        X_pair = np.vstack([data1, data2])
        y_pair = np.array([0] * len(data1) + [1] * len(data2))
        
        # Binary classifier for this pair
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_pair, y_pair)
        
        # Feature importance for this specific pair
        feature_importance_pair = pd.DataFrame({
            'feature': features_labels,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Statistical tests for each feature
        statistical_results = []
        for i, feature in enumerate(features_labels):
            vals1 = data1[:, i]
            vals2 = data2[:, i]
            
            # T-test
            t_stat, t_pvalue = ttest_ind(vals1, vals2)
            
            # Mann-Whitney U test
            u_stat, u_pvalue = mannwhitneyu(vals1, vals2, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(vals1) + np.var(vals2)) / 2)
            cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std if pooled_std > 0 else 0
            
            statistical_results.append({
                'feature': feature,
                'mean_diff': np.mean(vals1) - np.mean(vals2),
                'cohens_d': cohens_d,
                't_pvalue': t_pvalue,
                'u_pvalue': u_pvalue,
                'significant': t_pvalue < 0.05
            })
        
        stat_df = pd.DataFrame(statistical_results)
        stat_df = stat_df.sort_values('cohens_d', key=abs, ascending=False)
        
        # Cross-validation for this pair
        cv_scores = cross_val_score(clf, X_pair, y_pair, cv=5, scoring='accuracy')
        
        pair_results = {
            'classification_accuracy': cv_scores.mean(),
            'classification_std': cv_scores.std(),
            'feature_importance': feature_importance_pair,
            'statistical_tests': stat_df,
            'significant_features': stat_df[stat_df['significant']],
            'top_discriminative': stat_df.head(3)
        }
        
        pairwise_results[f"{pair[0]}_vs_{pair[1]}"] = pair_results
        
        print(f"Binary classification accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"Significant features: {len(stat_df[stat_df['significant']])}/{len(features_labels)}")
        print("Top 3 discriminative features:")
        for idx, row in stat_df.head(3).iterrows():
            print(f"  {row['feature']}: Cohen's d = {row['cohens_d']:.3f}, p = {row['t_pvalue']:.4f}")
    
    return pairwise_results

def plot_pairwise_discrimination_results(pairwise_results, features_labels, norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, plots_base_dir="loss_landscape_plots"):
    """
    Comprehensive visualization of pairwise discrimination results.
    """
    pairs = list(pairwise_results.keys())
    n_pairs = len(pairs)
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Classification accuracy comparison
    ax1 = plt.subplot(3, 4, 1)
    accuracies = [pairwise_results[pair]['classification_accuracy'] for pair in pairs]
    stds = [pairwise_results[pair]['classification_std'] for pair in pairs]
    
    bars = ax1.bar(range(n_pairs), accuracies, yerr=stds, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_xticks(range(n_pairs))
    ax1.set_xticklabels([pair.replace('_', ' ').title() for pair in pairs], rotation=45)
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Pairwise Classification Performance', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, (bar, acc, std) in enumerate(zip(bars, accuracies, stds)):
        ax1.text(bar.get_x() + bar.get_width()/2., acc + std + 0.02,
                f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2-4. Feature importance for each pair
    for i, pair in enumerate(pairs):
        ax = plt.subplot(3, 4, i + 2)
        
        feat_imp = pairwise_results[pair]['feature_importance'].head(8)  # Top 8 features
        
        bars = ax.barh(range(len(feat_imp)), feat_imp['importance'], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(feat_imp))))
        ax.set_yticks(range(len(feat_imp)))
        ax.set_yticklabels(feat_imp['feature'], fontsize=9)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top Features: {pair.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # 5. Effect sizes heatmap
    ax5 = plt.subplot(3, 4, (5, 6))
    
    # Create effect size matrix
    effect_matrix = np.zeros((len(features_labels), len(pairs)))
    for i, pair in enumerate(pairs):
        stat_tests = pairwise_results[pair]['statistical_tests']
        for j, feature in enumerate(features_labels):
            effect_size = stat_tests[stat_tests['feature'] == feature]['cohens_d'].iloc[0]
            effect_matrix[j, i] = abs(effect_size)
    
    im = ax5.imshow(effect_matrix, cmap='Reds', aspect='auto')
    ax5.set_xticks(range(len(pairs)))
    ax5.set_xticklabels([pair.replace('_', ' ').title() for pair in pairs])
    ax5.set_yticks(range(len(features_labels)))
    ax5.set_yticklabels(features_labels, fontsize=9)
    ax5.set_title('Effect Sizes (|Cohen\'s d|) Heatmap', fontweight='bold')
    
    # Add text annotations for high effect sizes
    for i in range(len(features_labels)):
        for j in range(len(pairs)):
            if effect_matrix[i, j] > 0.5:  # Only show large effects
                ax5.text(j, i, f'{effect_matrix[i, j]:.2f}',
                        ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im, ax=ax5, shrink=0.8, label='|Effect Size|')
    
    # 6. Significant features count
    ax6 = plt.subplot(3, 4, 7)
    
    sig_counts = [len(pairwise_results[pair]['significant_features']) for pair in pairs]
    bars = ax6.bar(range(n_pairs), sig_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax6.set_xticks(range(n_pairs))
    ax6.set_xticklabels([pair.replace('_', ' ').title() for pair in pairs], rotation=45)
    ax6.set_ylabel('Number of Significant Features')
    ax6.set_title('Significant Features (p<0.05)', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, sig_counts):
        ax6.text(bar.get_x() + bar.get_width()/2., count + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Distribution comparison for most discriminative feature
    ax7 = plt.subplot(3, 4, 8)
    
    # Find the feature with highest average effect size across all pairs
    avg_effects = {}
    for feature in features_labels:
        effects = []
        for pair in pairs:
            stat_tests = pairwise_results[pair]['statistical_tests']
            effect = abs(stat_tests[stat_tests['feature'] == feature]['cohens_d'].iloc[0])
            effects.append(effect)
        avg_effects[feature] = np.mean(effects)
    
    best_feature = max(avg_effects, key=avg_effects.get)
    best_feature_idx = features_labels.index(best_feature)
    
    # Plot distributions for the best discriminative feature
    datasets = {'forget': norm_forget_tensor, 'retain': norm_retain_tensor, 'holdout': norm_holdout_tensor}
    colors = {'forget': '#FF6B6B', 'retain': '#4ECDC4', 'holdout': '#45B7D1'}
    
    for dataset_name, tensor in datasets.items():
        values = tensor[:, best_feature_idx].numpy()
        ax7.hist(values, bins=20, alpha=0.6, label=dataset_name, color=colors[dataset_name], density=True)
    
    ax7.set_xlabel(f'{best_feature} Values')
    ax7.set_ylabel('Density')
    ax7.set_title(f'Distribution: {best_feature}\n(Most Discriminative Feature)', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plots_base_dir}/pairwise_discrimination_comprehensive.png', bbox_inches='tight', dpi=150)
    plt.show()
    
    return best_feature, avg_effects

def create_ill_discrimination_summary_report(ill_discrimination, boundary_results, pairwise_results, 
                                            best_discriminative_feature, feature_discrimination_scores, plots_base_dir="loss_landscape_plots"):
    """
    Generate a comprehensive summary report for ILL-based dataset discrimination.
    """
    print("="*80)
    print("COMPREHENSIVE INPUT LOSS LANDSCAPE (ILL) DISCRIMINATION REPORT")
    print("="*80)
    
    # Overall discrimination performance
    print("\n1. OVERALL DISCRIMINATION PERFORMANCE:")
    print("-" * 50)
    overall_cv = ill_discrimination['cv_scores'].mean()
    overall_std = ill_discrimination['cv_scores'].std()
    print(f"3-way classification (Forget/Retain/Holdout): {overall_cv:.3f} ± {overall_std:.3f}")
    
    print("\nClassifier comparison:")
    for clf_name, results in boundary_results.items():
        print(f"  {clf_name:15}: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    # Feature importance summary
    print("\n2. MOST DISCRIMINATIVE ILL FEATURES:")
    print("-" * 50)
    top_features = ill_discrimination['feature_importance'].head(5)
    for idx, row in top_features.iterrows():
        print(f"{row['rank']}. {row['feature']:20}: {row['importance']:.4f}")
    
    # Feature category analysis
    print("\n3. FEATURE CATEGORY DISCRIMINATION POWER:")
    print("-" * 50)
    feature_categories = {
        'Loss-based': ['original_loss', 'mean_neighbor_loss', 'max_neighbor_loss', 'min_neighbor_loss'],
        'Variance-based': ['loss_variance', 'loss_std', 'increment_variance'],
        'Gradient-based': ['mean_gradient', 'max_gradient', 'gradient_variance'],
        'Landscape': ['loss_volatility', 'local_curvature'],
        'Increment-based': ['mean_loss_increment', 'max_loss_increment', 'min_loss_increment']
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_features = ill_discrimination['feature_importance'][
            ill_discrimination['feature_importance']['feature'].isin(features)
        ]
        category_importance[category] = category_features['importance'].sum()
    
    sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    for category, importance in sorted_categories:
        print(f"  {category:15}: {importance:.4f}")
    
    # Pairwise discrimination
    print("\n4. PAIRWISE DISCRIMINATION RESULTS:")
    print("-" * 50)
    for pair_name, results in pairwise_results.items():
        acc = results['classification_accuracy']
        std = results['classification_std']
        n_sig = len(results['significant_features'])
        print(f"{pair_name.replace('_', ' ').title():20}: {acc:.3f}±{std:.3f} ({n_sig} significant features)")
    
    # Best discriminative features
    print(f"\n5. MOST DISCRIMINATIVE FEATURE OVERALL:")
    print("-" * 50)
    print(f"Feature: {best_discriminative_feature}")
    print(f"Average effect size across all pairs: {feature_discrimination_scores[best_discriminative_feature]:.3f}")
    
    # Recommendations
    print("\n6. RECOMMENDATIONS FOR ILL-BASED DISCRIMINATION:")
    print("-" * 50)
    
    best_pair = max(pairwise_results.items(), key=lambda x: x[1]['classification_accuracy'])
    worst_pair = min(pairwise_results.items(), key=lambda x: x[1]['classification_accuracy'])
    
    print(f"• Easiest to distinguish: {best_pair[0].replace('_', ' ')} ({best_pair[1]['classification_accuracy']:.3f})")
    print(f"• Hardest to distinguish: {worst_pair[0].replace('_', ' ')} ({worst_pair[1]['classification_accuracy']:.3f})")
    
    print(f"• Focus on top 3 features: {', '.join(top_features.head(3)['feature'].tolist())}")
    
    most_important_category = sorted_categories[0][0]
    print(f"• Most important feature category: {most_important_category}")
    
    # Best classifier
    best_classifier = max(boundary_results.items(), key=lambda x: x[1]['cv_mean'])
    print(f"• Best performing classifier: {best_classifier[0]} ({best_classifier[1]['cv_mean']:.3f})")
    
    print("\n" + "="*80)
    
    # Save comprehensive results
    comprehensive_ill_results = {
        'discrimination_analysis': ill_discrimination,
        'boundary_analysis': boundary_results,
        'pairwise_analysis': pairwise_results,
        'best_discriminative_feature': best_discriminative_feature,
        'feature_discrimination_scores': feature_discrimination_scores,
        'feature_categories': feature_categories,
        'category_importance': category_importance
    }
    
    results_file = f'{plots_base_dir}/comprehensive_ill_discrimination_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(comprehensive_ill_results, f)
    
    print(f"Complete ILL discrimination results saved to: {results_file}")
    
    return comprehensive_ill_results

def analyze_loss_landscape_categories(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, features_labels):
    """
    Analyze loss landscape features by semantic categories relevant to ILL.
    """
    # Define feature categories for ILL analysis
    feature_categories = {
        'Base Loss': ['original_loss', 'mean_neighbor_loss', 'max_neighbor_loss', 'min_neighbor_loss'],
        'Variability': ['loss_variance', 'loss_std', 'increment_variance'],
        'Gradient': ['mean_gradient', 'max_gradient', 'gradient_variance'],
        'Landscape': ['loss_volatility', 'local_curvature'],
        'Increment': ['mean_loss_increment', 'max_loss_increment', 'min_loss_increment']
    }
    
    datasets = {
        'retain': norm_retain_tensor,
        'holdout': norm_holdout_tensor,
        'forget': norm_forget_tensor
    }
    
    category_analysis = {}
    
    # Analyze each category
    for category, features in feature_categories.items():
        category_data = {}
        
        for dataset_name, tensor in datasets.items():
            category_features = []
            for feature in features:
                if feature in features_labels:
                    feat_idx = features_labels.index(feature)
                    category_features.append(tensor[:, feat_idx].numpy())
            
            if category_features:
                category_tensor = np.column_stack(category_features)
                
                # Compute category-level statistics
                category_data[dataset_name] = {
                    'mean_values': np.mean(category_tensor, axis=0),
                    'std_values': np.std(category_tensor, axis=0),
                    'category_mean': np.mean(category_tensor),
                    'category_std': np.std(category_tensor),
                    'feature_correlations': np.corrcoef(category_tensor.T),
                    'feature_names': [f for f in features if f in features_labels]
                }
        
        category_analysis[category] = category_data
    
    return category_analysis

def plot_ill_category_analysis(category_analysis, plots_base_dir="loss_landscape_plots"):
    """
    Create comprehensive visualizations for ILL category analysis.
    """
    categories = list(category_analysis.keys())
    datasets = ['retain', 'holdout', 'forget']
    colors = {'retain': 'blue', 'holdout': 'green', 'forget': 'red'}
    
    # 1. Category-level means comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        category_means = []
        category_stds = []
        
        for dataset in datasets:
            if dataset in category_analysis[category]:
                category_means.append(category_analysis[category][dataset]['category_mean'])
                category_stds.append(category_analysis[category][dataset]['category_std'])
            else:
                category_means.append(0)
                category_stds.append(0)
        
        x_pos = np.arange(len(datasets))
        bars = ax.bar(x_pos, category_means, yerr=category_stds, 
                     color=[colors[d] for d in datasets], alpha=0.7, capsize=5)
        
        ax.set_title(f'{category} Features\nMean Values Across Datasets', fontweight='bold')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Mean Feature Value')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(datasets)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, category_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Hide unused subplot
    if len(categories) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{plots_base_dir}/ill_category_means.png', bbox_inches='tight', dpi=150)
    plt.show()
    
    # 2. Feature-level heatmap for each category
    for category in categories:
        if category not in category_analysis:
            continue
            
        # Get feature names for this category
        feature_names = None
        for dataset in datasets:
            if dataset in category_analysis[category]:
                feature_names = category_analysis[category][dataset]['feature_names']
                break
        
        if not feature_names:
            continue
            
        # Create data matrix
        data_matrix = []
        for feature_idx in range(len(feature_names)):
            row = []
            for dataset in datasets:
                if dataset in category_analysis[category]:
                    row.append(category_analysis[category][dataset]['mean_values'][feature_idx])
                else:
                    row.append(0)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names)*0.8)))
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        
        ax.set_title(f'{category} Features: Dataset Comparison\nHigher Values = More Loss/Gradient Activity', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Features')
        ax.set_xticks(range(len(datasets)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(datasets)
        ax.set_yticklabels(feature_names)
        
        # Add text annotations
        for i in range(len(feature_names)):
            for j in range(len(datasets)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.4f}',
                              ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(f'{plots_base_dir}/ill_{category.lower()}_features.png', bbox_inches='tight', dpi=150)
        plt.show()

def analyze_semantic_embeddings(forget_data, holdout_data, retain_data):
    """
    Analyze sentence embeddings to understand semantic patterns in loss landscapes.
    """
    from sentence_transformers import SentenceTransformer
    
    # Load a sentence transformer model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Using SentenceTransformer for semantic analysis...")
    except:
        print("SentenceTransformer not available. Install with: pip install sentence-transformers")
        return None
    
    # Extract texts from data
    datasets = {
        'forget': [item['text'] for item in forget_data if item.get('valid_example', True)],
        'holdout': [item['text'] for item in holdout_data if item.get('valid_example', True)],
        'retain': [item['text'] for item in retain_data if item.get('valid_example', True)]
    }
    
    # Limit to reasonable sample size for computation
    max_samples = 500
    for dataset_name in datasets:
        if len(datasets[dataset_name]) > max_samples:
            datasets[dataset_name] = datasets[dataset_name][:max_samples]
            print(f"Limiting {dataset_name} to {max_samples} samples for embedding analysis")
    
    # Compute embeddings
    embedding_results = {}
    
    for dataset_name, texts in datasets.items():
        print(f"Computing embeddings for {dataset_name} dataset ({len(texts)} samples)...")
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        
        embedding_results[dataset_name] = {
            'embeddings': embeddings,
            'texts': texts,
            'embedding_mean': np.mean(embeddings, axis=0),
            'embedding_std': np.std(embeddings, axis=0),
            'centroid': np.mean(embeddings, axis=0),
            'intra_cluster_distance': np.mean(pdist(embeddings))
        }
    
    return embedding_results
