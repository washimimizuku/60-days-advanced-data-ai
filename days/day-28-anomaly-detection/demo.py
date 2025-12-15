#!/usr/bin/env python3
"""
Interactive Anomaly Detection Demo
Demonstrates all anomaly detection methods with real data examples
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_datasets():
    """Create comprehensive sample datasets with known anomalies"""
    
    print("📊 Creating sample anomaly detection datasets...")
    
    # Financial transactions dataset
    np.random.seed(42)
    n_normal = 9000
    n_anomaly = 1000
    
    # Normal transactions
    normal_amounts = np.random.lognormal(mean=3, sigma=1, size=n_normal)
    normal_amounts = np.clip(normal_amounts, 1, 1000)
    
    normal_times = np.random.choice(range(6, 22), size=n_normal)  # Business hours
    normal_locations = np.random.choice([0, 1], size=n_normal, p=[0.95, 0.05])  # Mostly domestic
    
    # Anomalous transactions
    anomaly_amounts = np.random.uniform(2000, 10000, size=n_anomaly)  # High amounts
    anomaly_times = np.random.choice(range(24), size=n_anomaly)  # Any time
    anomaly_locations = np.random.choice([0, 1], size=n_anomaly, p=[0.3, 0.7])  # More international
    
    # Combine datasets
    amounts = np.concatenate([normal_amounts, anomaly_amounts])
    times = np.concatenate([normal_times, anomaly_times])
    locations = np.concatenate([normal_locations, anomaly_locations])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    financial_data = pd.DataFrame({
        'amount': amounts,
        'hour': times,
        'is_international': locations,
        'is_anomaly': labels.astype(bool)
    })
    
    # Network traffic dataset
    np.random.seed(123)
    n_normal_net = 8000
    n_anomaly_net = 2000
    
    # Normal network traffic
    normal_packet_sizes = np.random.exponential(scale=500, size=n_normal_net)
    normal_packet_sizes = np.clip(normal_packet_sizes, 64, 1500)
    normal_durations = np.random.exponential(scale=30, size=n_normal_net)
    normal_ports = np.random.choice([80, 443, 22, 21], size=n_normal_net, p=[0.4, 0.3, 0.2, 0.1])
    
    # Anomalous network traffic (DDoS, port scans)
    anomaly_packet_sizes = np.random.choice([64, 1500], size=n_anomaly_net, p=[0.7, 0.3])
    anomaly_durations = np.random.exponential(scale=1, size=n_anomaly_net)  # Very short
    anomaly_ports = np.random.randint(1, 1024, size=n_anomaly_net)  # Random ports
    
    # Combine network data
    packet_sizes = np.concatenate([normal_packet_sizes, anomaly_packet_sizes])
    durations = np.concatenate([normal_durations, anomaly_durations])
    ports = np.concatenate([normal_ports, anomaly_ports])
    net_labels = np.concatenate([np.zeros(n_normal_net), np.ones(n_anomaly_net)])
    
    network_data = pd.DataFrame({
        'packet_size': packet_sizes,
        'duration': durations,
        'port': ports,
        'is_anomaly': net_labels.astype(bool)
    })
    
    # Time series sensor data
    dates = pd.date_range('2024-01-01', periods=1440, freq='1min')  # 1 day of minute data
    
    # Normal sensor readings with daily pattern
    base_temp = 20 + 5 * np.sin(2 * np.pi * np.arange(1440) / 1440)  # Daily cycle
    temperature = base_temp + np.random.normal(0, 1, 1440)
    
    # Inject temperature anomalies
    anomaly_indices = np.random.choice(1440, 50, replace=False)
    temperature[anomaly_indices] += np.random.uniform(15, 25, 50)  # Temperature spikes
    
    sensor_labels = np.zeros(1440, dtype=bool)
    sensor_labels[anomaly_indices] = True
    
    sensor_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature,
        'is_anomaly': sensor_labels
    })
    
    return {
        'financial': financial_data,
        'network': network_data,
        'sensor': sensor_data
    }

def demo_statistical_methods(data, name):
    """Demonstrate statistical anomaly detection methods"""
    
    print(f"\n🔧 Statistical Methods Demo: {name}")
    print("-" * 50)
    
    # Prepare numeric features
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'is_anomaly']
    
    if len(numeric_cols) == 0:
        print("❌ No numeric features found")
        return
    
    X = data[numeric_cols].values
    y_true = data['is_anomaly'].values if 'is_anomaly' in data.columns else None
    
    # Z-score method
    print("Testing Z-score method...")
    z_scores = np.abs(stats.zscore(X, axis=0))
    z_anomalies = (z_scores > 3).any(axis=1)
    
    # IQR method
    print("Testing IQR method...")
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_anomalies = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
    
    # Modified Z-score (MAD)
    print("Testing Modified Z-score method...")
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad[mad == 0] = 1e-6  # Avoid division by zero
    modified_z_scores = 0.6745 * (X - median) / mad
    mad_anomalies = (np.abs(modified_z_scores) > 3.5).any(axis=1)
    
    # Evaluate if ground truth available
    if y_true is not None:
        methods = {
            'Z-score': z_anomalies,
            'IQR': iqr_anomalies,
            'Modified Z-score': mad_anomalies
        }
        
        print("\n📊 Statistical Methods Performance:")
        for method_name, predictions in methods.items():
            precision = precision_score(y_true, predictions)
            recall = recall_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)
            print(f"  {method_name:15} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Visualization
    if X.shape[1] >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        methods_viz = [
            ('Z-score', z_anomalies),
            ('IQR', iqr_anomalies),
            ('Modified Z-score', mad_anomalies)
        ]
        
        for i, (method_name, anomalies) in enumerate(methods_viz):
            ax = axes[i]
            
            # Plot normal points
            normal_mask = ~anomalies
            ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                      c='blue', alpha=0.6, s=20, label='Normal')
            
            # Plot anomalies
            ax.scatter(X[anomalies, 0], X[anomalies, 1], 
                      c='red', alpha=0.8, s=30, label='Anomaly')
            
            ax.set_title(f'{method_name}\n{np.sum(anomalies)} anomalies detected')
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Statistical Anomaly Detection: {name}')
        plt.tight_layout()
        plt.show()
    
    print("✅ Statistical methods demo completed")

def demo_ml_methods(data, name):
    """Demonstrate ML-based anomaly detection methods"""
    
    print(f"\n🤖 ML Methods Demo: {name}")
    print("-" * 50)
    
    # Prepare features
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'is_anomaly']
    
    if len(numeric_cols) == 0:
        print("❌ No numeric features found")
        return
    
    X = data[numeric_cols].values
    y_true = data['is_anomaly'].values if 'is_anomaly' in data.columns else None
    
    # Split data for training (use normal data for unsupervised methods)
    if y_true is not None:
        normal_mask = ~y_true
        X_train = X[normal_mask]
        X_test = X
        y_test = y_true
    else:
        X_train = X
        X_test = X
        y_test = None
    
    # Isolation Forest
    print("Testing Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train)
    iso_predictions = iso_forest.predict(X_test)
    iso_scores = iso_forest.decision_function(X_test)
    iso_anomalies = iso_predictions == -1
    
    # One-Class SVM
    print("Testing One-Class SVM...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    oc_svm = OneClassSVM(kernel='rbf', nu=0.1)
    oc_svm.fit(X_train_scaled)
    svm_predictions = oc_svm.predict(X_test_scaled)
    svm_scores = oc_svm.decision_function(X_test_scaled)
    svm_anomalies = svm_predictions == -1
    
    # Ensemble method (simple voting)
    print("Testing Ensemble method...")
    ensemble_votes = iso_anomalies.astype(int) + svm_anomalies.astype(int)
    ensemble_anomalies = ensemble_votes >= 1  # At least one method agrees
    
    # Evaluate performance
    if y_test is not None:
        methods = {
            'Isolation Forest': iso_anomalies,
            'One-Class SVM': svm_anomalies,
            'Ensemble': ensemble_anomalies
        }
        
        print("\n📊 ML Methods Performance:")
        for method_name, predictions in methods.items():
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            print(f"  {method_name:15} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Visualization
    if X.shape[1] >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        methods_viz = [
            ('Isolation Forest', iso_anomalies, iso_scores),
            ('One-Class SVM', svm_anomalies, svm_scores),
            ('Ensemble', ensemble_anomalies, None)
        ]
        
        for i, (method_name, anomalies, scores) in enumerate(methods_viz):
            ax = axes[i]
            
            if scores is not None:
                # Color by anomaly score
                scatter = ax.scatter(X[:, 0], X[:, 1], c=scores, 
                                   cmap='RdYlBu_r', alpha=0.6, s=20)
                plt.colorbar(scatter, ax=ax, label='Anomaly Score')
            else:
                # Color by anomaly flag
                normal_mask = ~anomalies
                ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                          c='blue', alpha=0.6, s=20, label='Normal')
                ax.scatter(X[anomalies, 0], X[anomalies, 1], 
                          c='red', alpha=0.8, s=30, label='Anomaly')
                ax.legend()
            
            ax.set_title(f'{method_name}\n{np.sum(anomalies)} anomalies detected')
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'ML Anomaly Detection: {name}')
        plt.tight_layout()
        plt.show()
    
    print("✅ ML methods demo completed")

def demo_time_series_methods(data, name):
    """Demonstrate time series anomaly detection"""
    
    print(f"\n📈 Time Series Methods Demo: {name}")
    print("-" * 50)
    
    if 'timestamp' not in data.columns:
        print("❌ No timestamp column found for time series analysis")
        return
    
    # Prepare time series data
    ts_data = data.set_index('timestamp').sort_index()
    
    # Get numeric columns
    numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'is_anomaly']
    
    if len(numeric_cols) == 0:
        print("❌ No numeric features found")
        return
    
    # Use first numeric column for demo
    ts_col = numeric_cols[0]
    ts = ts_data[ts_col]
    y_true = ts_data['is_anomaly'].values if 'is_anomaly' in ts_data.columns else None
    
    # Statistical Process Control (SPC)
    print("Testing Statistical Process Control...")
    window_size = 30
    rolling_mean = ts.rolling(window=window_size).mean()
    rolling_std = ts.rolling(window=window_size).std()
    
    upper_limit = rolling_mean + 3 * rolling_std
    lower_limit = rolling_mean - 3 * rolling_std
    
    spc_anomalies = (ts > upper_limit) | (ts < lower_limit)
    spc_anomalies = spc_anomalies.fillna(False)
    
    # Moving Z-score
    print("Testing Moving Z-score...")
    rolling_z = (ts - rolling_mean) / rolling_std
    moving_z_anomalies = np.abs(rolling_z) > 3
    moving_z_anomalies = moving_z_anomalies.fillna(False)
    
    # Evaluate performance
    if y_true is not None:
        methods = {
            'SPC': spc_anomalies.values,
            'Moving Z-score': moving_z_anomalies.values
        }
        
        print("\n📊 Time Series Methods Performance:")
        for method_name, predictions in methods.items():
            precision = precision_score(y_true, predictions)
            recall = recall_score(y_true, predictions)
            f1 = f1_score(y_true, predictions)
            print(f"  {method_name:15} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # SPC Chart
    ax1 = axes[0]
    ax1.plot(ts.index, ts.values, 'b-', alpha=0.7, label='Data')
    ax1.plot(ts.index, rolling_mean, 'g--', label='Mean')
    ax1.fill_between(ts.index, lower_limit, upper_limit, alpha=0.2, color='gray', label='Control Limits')
    
    if y_true is not None:
        anomaly_times = ts.index[y_true]
        ax1.scatter(anomaly_times, ts[y_true], c='red', s=50, label='True Anomalies', zorder=5)
    
    spc_anomaly_times = ts.index[spc_anomalies]
    ax1.scatter(spc_anomaly_times, ts[spc_anomalies], c='orange', s=30, 
               marker='x', label='SPC Detected', zorder=4)
    
    ax1.set_title('Statistical Process Control')
    ax1.set_ylabel(ts_col)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Moving Z-score
    ax2 = axes[1]
    ax2.plot(ts.index, rolling_z, 'b-', alpha=0.7, label='Moving Z-score')
    ax2.axhline(y=3, color='r', linestyle='--', alpha=0.7, label='Threshold')
    ax2.axhline(y=-3, color='r', linestyle='--', alpha=0.7)
    
    moving_z_anomaly_times = ts.index[moving_z_anomalies]
    ax2.scatter(moving_z_anomaly_times, rolling_z[moving_z_anomalies], 
               c='orange', s=30, marker='x', label='Detected Anomalies', zorder=4)
    
    ax2.set_title('Moving Z-score Anomaly Detection')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Z-score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Time Series Anomaly Detection: {name}')
    plt.tight_layout()
    plt.show()
    
    print("✅ Time series methods demo completed")

def demo_performance_comparison(datasets):
    """Compare performance across all methods and datasets"""
    
    print(f"\n📊 Performance Comparison Across All Methods")
    print("-" * 60)
    
    results = []
    
    for dataset_name, data in datasets.items():
        if 'is_anomaly' not in data.columns:
            continue
        
        print(f"\nEvaluating {dataset_name} dataset...")
        
        # Prepare features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'is_anomaly']
        
        if len(numeric_cols) == 0:
            continue
        
        X = data[numeric_cols].values
        y_true = data['is_anomaly'].values
        
        # Normal data for training
        normal_mask = ~y_true
        X_train = X[normal_mask]
        
        # Test all methods
        methods_to_test = {}
        
        # Statistical methods
        try:
            z_scores = np.abs(stats.zscore(X, axis=0))
            methods_to_test['Z-score'] = (z_scores > 3).any(axis=1)
        except:
            pass
        
        try:
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            methods_to_test['IQR'] = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
        except:
            pass
        
        # ML methods
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X_train)
            methods_to_test['Isolation Forest'] = iso_forest.predict(X) == -1
        except:
            pass
        
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_scaled = scaler.transform(X)
            
            oc_svm = OneClassSVM(kernel='rbf', nu=0.1)
            oc_svm.fit(X_train_scaled)
            methods_to_test['One-Class SVM'] = oc_svm.predict(X_scaled) == -1
        except:
            pass
        
        # Evaluate each method
        for method_name, predictions in methods_to_test.items():
            try:
                precision = precision_score(y_true, predictions)
                recall = recall_score(y_true, predictions)
                f1 = f1_score(y_true, predictions)
                
                results.append({
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })
            except:
                continue
    
    # Create results DataFrame and visualize
    if results:
        results_df = pd.DataFrame(results)
        
        # Print summary table
        print("\n📋 Performance Summary:")
        pivot_table = results_df.pivot_table(
            values=['Precision', 'Recall', 'F1-Score'], 
            index='Method', 
            columns='Dataset', 
            aggfunc='mean'
        )
        print(pivot_table.round(3))
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        for i, metric in enumerate(metrics):
            metric_data = results_df.pivot(index='Method', columns='Dataset', values=metric)
            
            sns.heatmap(metric_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=axes[i], cbar_kws={'label': metric})
            axes[i].set_title(f'{metric} by Method and Dataset')
            axes[i].set_xlabel('Dataset')
            axes[i].set_ylabel('Method')
        
        plt.tight_layout()
        plt.show()
        
        # Best method per dataset
        print("\n🏆 Best Methods by Dataset:")
        for dataset in results_df['Dataset'].unique():
            dataset_results = results_df[results_df['Dataset'] == dataset]
            best_method = dataset_results.loc[dataset_results['F1-Score'].idxmax()]
            print(f"  {dataset:10} - {best_method['Method']:15} (F1: {best_method['F1-Score']:.3f})")
    
    print("✅ Performance comparison completed")

def main():
    """Main demo function"""
    
    print("🚀 Anomaly Detection Interactive Demo")
    print("=" * 60)
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    for name, data in datasets.items():
        print(f"\n📊 Dataset: {name}")
        print(f"   Shape: {data.shape}")
        print(f"   Anomalies: {data['is_anomaly'].sum()} ({data['is_anomaly'].mean():.2%})")
        
        # Plot dataset overview
        if name != 'sensor':  # Skip time series for overview
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'is_anomaly']
            
            if len(numeric_cols) >= 2:
                plt.figure(figsize=(8, 6))
                
                normal_data = data[~data['is_anomaly']]
                anomaly_data = data[data['is_anomaly']]
                
                plt.scatter(normal_data[numeric_cols[0]], normal_data[numeric_cols[1]], 
                           c='blue', alpha=0.6, s=20, label='Normal')
                plt.scatter(anomaly_data[numeric_cols[0]], anomaly_data[numeric_cols[1]], 
                           c='red', alpha=0.8, s=30, label='Anomaly')
                
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
                plt.title(f'{name.title()} Dataset Overview')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
        
        # Run method demos
        demo_statistical_methods(data, name)
        demo_ml_methods(data, name)
        
        if name == 'sensor':  # Time series specific
            demo_time_series_methods(data, name)
    
    # Performance comparison
    demo_performance_comparison(datasets)
    
    print("\n🎯 Demo Summary:")
    print("✅ Statistical Methods: Simple, interpretable, good for well-behaved data")
    print("✅ ML Methods: Handle complex patterns, robust to data distribution")
    print("✅ Time Series Methods: Account for temporal dependencies and trends")
    print("✅ Ensemble Methods: Combine strengths of multiple approaches")
    
    print("\n💡 Key Insights:")
    print("• No single method works best for all scenarios")
    print("• Ensemble approaches often provide more robust results")
    print("• Domain knowledge is crucial for threshold setting")
    print("• Time series data requires specialized techniques")
    print("• Evaluation metrics should align with business objectives")
    
    print("\n🚀 Ready for production anomaly detection systems!")

if __name__ == "__main__":
    main()