import matplotlib.pyplot as plt
import seaborn as sns

from src.config.paths import get_plot_path


def summarize_dataset(df, numerical_features, categorical_features):
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe(include='all'))

    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    for column in categorical_features:
        print(f"\n{column} Value Counts:")
        print(df[column].value_counts())

    print("\nNull Value Summary:")
    print(df.isnull().sum().sort_values(ascending=False))

    print("\nTransposed Summary Statistics:")
    print(df.describe().T)


def plot_distribution(data, features, plot_type='hist', figsize=(26, 4)):
    """
    Create distribution plots for specified numerical features

    Args:
        data: pandas DataFrame containing the data
        features: list of numerical column names to plot
        plot_type: str, type of plot to create ('hist', 'kde', or 'box')
        figsize: tuple specifying figure dimensions
    """
    if plot_type == 'box' or plot_type == 'violin':
        figsize = (26, 8)  # Boxplots typically need more vertical space

    _, ax = plt.subplots(nrows=1, ncols=len(features), figsize=figsize)

    plot_funcs = {
        'hist': sns.histplot,
        'kde': sns.kdeplot,
        'box': sns.boxplot,
        'violin': sns.violinplot
    }

    if plot_type not in plot_funcs:
        raise ValueError("plot_type must be one of: 'hist', 'kde', 'box'")

    plot_func = plot_funcs[plot_type]

    for index, col in enumerate(features):
        if plot_type == 'box':
            plot_func(data=data, y=col, ax=ax[index])
        elif plot_type == 'violin':
            plot_func(data=data, y=col, ax=ax[index], inner='quartile')
        else:
            plot_func(data=data, x=col, ax=ax[index])
        ax[index].set_title(f'{col} distribution')

    # Save the figure after all features are plotted
    plt.tight_layout()
    plt.savefig(get_plot_path(f'{plot_type}_distributions', 'png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()


def plot_categorical_distribution(data, features, plot_type='count', figsize=None):
    """
    Create distribution plots for categorical features

    Parameters:
    -----------
    data : pandas DataFrame
        The input DataFrame containing the features
    features : list
        List of categorical column names to plot
    plot_type : str, optional (default='count')
        Type of plot to create. Options: 'count' or 'pie'
    figsize : tuple, optional
        Figure size in inches (width, height). If None, will be calculated based on features
    """
    # Filter out identifier columns
    features = [c for c in features if c not in [
        'Item_Identifier', 'Outlet_Identifier']]

    # Calculate layout
    n_features = len(features)
    # Ceiling division for odd number of features
    n_rows = (n_features + 1) // 2

    # Set default figsize if not provided
    if figsize is None:
        if plot_type == 'count':
            figsize = (32, 12*n_rows)
        else:  # pie
            figsize = (16, 16)

    # Create subplots
    _, ax = plt.subplots(nrows=n_rows, ncols=2, figsize=figsize)

    # Make ax 2D even if there's only one row
    if n_rows == 1:
        ax = ax.reshape(1, -1)

    # Plot each feature
    for index, col in enumerate(features):
        r = index // 2
        c = index % 2

        if plot_type == 'count':
            g = sns.countplot(data=data, x=col, ax=ax[r][c], width=0.6)
            g.set_xticklabels(g.get_xticklabels(),
                              rotation=45, ha="right", fontsize=18)
            ax[r][c].set_title(f'{col} distribution', fontsize=24)
        else:  # pie
            data[col].value_counts().plot(
                kind="pie", autopct="%.2f", ax=ax[r][c])

    # Hide empty subplots if odd number of features
    if n_features % 2 == 1:
        ax[n_rows-1][1].set_visible(False)

    plt.tight_layout()
    # Save plot
    plt.savefig(get_plot_path(
        f'categorical_{plot_type}_plots', 'png'), bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(data):
    """
    Create correlation heatmap for numerical features

    Args:
        data (pd.DataFrame): Input dataframe
    """
    # Create correlation matrix only for numerical columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numerical_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Correlation Heatmap of Numerical Features')

    plt.tight_layout()
    plt.savefig(get_plot_path('correlation_heatmap', 'png'),
                bbox_inches='tight')
    plt.close()
