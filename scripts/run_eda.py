import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.load_data import load_data
from src.features.build_features import separate_features
from src.eda.explore import plot_correlation_heatmap, summarize_dataset, plot_distribution, plot_categorical_distribution
from src.utils.utils import plot_barplot, plot_boxplot, plot_scatter


if __name__ == "__main__":
    train, test = load_data()
    numerical_features, categorical_features, target = separate_features(train)

    summarize_dataset(train, numerical_features, categorical_features)

    # Create a list that includes numerical features and target
    features_with_target = numerical_features + [target]
    
    plot_distribution(train, features_with_target, plot_type='hist')
    plot_distribution(train, features_with_target, plot_type='kde')
    plot_distribution(train, features_with_target, plot_type='box')
    plot_distribution(train, features_with_target, plot_type='violin')

    plot_categorical_distribution(
        train, categorical_features, plot_type='count', figsize=None)
    plot_categorical_distribution(
        train, categorical_features, plot_type='pie', figsize=None)

    plot_scatter(train, numerical_features, target)
    plot_scatter(train, numerical_features, target, hue='Outlet_Type')

    plot_correlation_heatmap(train)

    plot_barplot(train, 'Outlet_Type', target)
    plot_barplot(train, 'Outlet_Size', target)
