import matplotlib.pyplot as plt
import seaborn as sns

from ..config.paths import get_plot_path


def detect_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return upper_limit, lower_limit


def save_object(obj, filepath):
    """
    Save any object to pickle file

    Args:
        obj: Object to save
        filepath (str): Path to save the object
    """
    import pickle
    import os

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filepath):
    """
    Load any object from pickle file

    Args:
        filepath (str): Path to the object file

    Returns:
        The loaded object
    """
    import pickle

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    return obj


def plot_barplot(data, x_feature, y_feature, title=None, figsize=(10, 6), save_path=None):
    """
    Create a generic bar plot showing mean y_feature value for each category in x_feature

    Args:
        data (pd.DataFrame): Input dataframe
        x_feature (str): Name of categorical feature for x-axis
        y_feature (str): Name of numerical feature for y-axis
        title (str, optional): Plot title. If None, will be auto-generated
        figsize (tuple): Figure size in inches (width, height)
        save_path (str, optional): Path to save the plot. If None, won't save
    """

    plt.figure(figsize=figsize)
    sns.barplot(data=data, x=x_feature, y=y_feature)

    if title is None:
        title = f'Average {y_feature} by {x_feature}'
    plt.title(title)

    plt.tight_layout()

    if save_path is None:
        save_path = get_plot_path(f'{x_feature}_barplot', 'png')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_scatter(data, x_features, y_feature, hue=None, figsize=(26, 4), save_path=None, title=None):
    """
    Create scatter plots for multiple x_features against y_feature

    Args:
        data (pd.DataFrame): Input dataframe
        x_features (list): List of numerical feature names for x-axis
        y_feature (str): Name of target/numerical feature for y-axis
        hue (str, optional): Categorical variable for color coding
        figsize (tuple): Figure size in inches (width, height)
        save_path (str, optional): Path to save the plot. If None, won't save
        title (str, optional): Plot title. If None, will be auto-generated
    """

    _, ax = plt.subplots(nrows=1, ncols=len(x_features), figsize=figsize)

    # Handle single feature case
    if len(x_features) == 1:
        ax = [ax]

    for index, col in enumerate(x_features):
        if hue:
            sns.scatterplot(data=data, x=col, y=y_feature,
                            hue=hue, ax=ax[index])
            ax[index].set_title(f'{col} vs {y_feature} by {hue}')
        else:
            sns.scatterplot(data=data, x=col, y=y_feature, ax=ax[index])
            ax[index].set_title(f'{col} vs {y_feature}')

    plt.tight_layout()

    if save_path is None:
        save_path = get_plot_path(f'{x_features}_scatterplot', 'png')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_boxplot(data, x_feature, y_feature, figsize=(10, 6), save_path=None, title=None):
    """
    Create a box plot showing the distribution of y_feature for each category in x_feature

    Args:
        data (pd.DataFrame): Input dataframe
        x_feature (str): Name of categorical feature for x-axis
        y_feature (str): Name of numerical feature for y-axis
    """

    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x_feature, y=y_feature)

    if title is None:
        title = f'Box plot of {y_feature} by {x_feature}'
    plt.title(title)

    plt.tight_layout()

    if save_path is None:
        save_path = get_plot_path(f'{x_feature}_boxplot', 'png')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
