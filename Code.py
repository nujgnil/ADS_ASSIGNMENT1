import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

def load_data():
    # Load the data
    file_path = (r"C:/Users/Ling Jun/Desktop/PSB/Masters/Sem 1/Applied Data Science/Assignment/Assignment 1/archive/heart.csv")
    df = pd.read_csv(file_path)
    print(df.describe())
    print(df.head())    
    
    return df
# ========================== PRE-PROCCESSING: CLEANING DATA =====================================

def preprocessing(df, handle_missing='drop', handle_duplicates='drop', handle_outliers=None, categorical_threshold=10):
    """
    Preprocess a DataFrame by handling missing values, duplicates, outliers, and categorical data.

    Parameters:
    - df: Input DataFrame.
    - handle_missing: How to handle missing values. Options: 'drop', 'fill', or a dictionary for specific fill values.
    - handle_duplicates: How to handle duplicate rows. Options: 'drop' or 'keep'.
    - handle_outliers: How to handle outliers. Options: None, 'remove', or 'winsorize'.
    - categorical_threshold: Threshold for converting high-cardinality categorical columns to numerical.

    Returns:
    - df_cleaned: The preprocessed DataFrame.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_cleaned = df.copy()

    # Step 1: Handle invalid values in `Oldpeak` (negative values)
    print("Invalid `Oldpeak` values (negative values):")
    print(df_cleaned[df_cleaned['Oldpeak'] < 0])
    df_cleaned.loc[df_cleaned['Oldpeak'] < 0, 'Oldpeak'] = np.nan  # Replace negative values with NaN

    # Step 2: Handle invalid values in `RestingBP` (0 values)
    print("Invalid `RestingBP` values (0 values):")
    print(df_cleaned[df_cleaned['RestingBP'] == 0])
    df_cleaned.loc[df_cleaned['RestingBP'] == 0, 'RestingBP'] = np.nan  # Replace 0 values with NaN

    # Step 3: Handle missing values
    print("Checking for missing values: \n", df_cleaned.isnull().sum())
    if handle_missing == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif handle_missing == 'fill':
        df_cleaned = df_cleaned.fillna(method='ffill')  # Forward fill
    elif isinstance(handle_missing, dict):
        df_cleaned = df_cleaned.fillna(handle_missing)  # Fill with specific values

    # Step 4: Handle duplicates
    print("Checking for duplicate rows: \n", df_cleaned.duplicated().sum())
    if handle_duplicates == 'drop':
        df_cleaned = df_cleaned.drop_duplicates()

    # Step 5: Handle outliers
    print("Checking for outliers: \n", df_cleaned.describe())
    if handle_outliers == 'remove':
        # Remove outliers using IQR method
        numerical_features = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_features:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    elif handle_outliers == 'winsorize':
        # Winsorize outliers
        from scipy.stats.mstats import winsorize
        numerical_features = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_features:
            df_cleaned[col] = winsorize(df_cleaned[col], limits=[0.05, 0.05])
    '''
    # Step 6: Handle categorical data
    print("Checking for categorical data: \n", df_cleaned.dtypes)
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object' or df_cleaned[col].dtype.name == 'category':
            print(f"Column: {col}")
            print(df_cleaned[col].value_counts(), "\n")
            # Convert high-cardinality categorical columns to numerical if necessary
            if df_cleaned[col].nunique() > categorical_threshold:
                print(f"High-cardinality categorical column '{col}' detected. Consider encoding or dropping.")
            else:
                # Perform one-hot encoding for low-cardinality categorical columns
                df_cleaned = pd.get_dummies(df_cleaned, columns=[col], drop_first=True)
    '''
    # Step 7: Return the cleaned DataFrame
    return df_cleaned
# =================== PLOT ALL FEATURES AGAINST COUNT=============
def plot_categorical_features(df):
    """
    Plot count plots for categorical features with data labels.

    Parameters:
    - df: Input DataFrame.
    """
    # Identify categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    # Create subplots for categorical features
    plt.figure(figsize=(20, len(categorical_features) * 5))
    plot_index = 1

    for col in categorical_features:
        plt.subplot(len(categorical_features), 1, plot_index)
        ax = sns.countplot(x=col, data=df, palette='viridis')
        plt.title(f'Count Plot of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12)

        # Add data labels to bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2, p.get_height()), 
                        ha='center', va='bottom', fontsize=10, color='black')

        plot_index += 1

    # Adjust layout
    plt.suptitle('Distribution of Categorical Features', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_numerical_features(df):
    """
    Plot histograms for numerical features with data labels.

    Parameters:
    - df: Input DataFrame.
    """
    # Identify numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

    # Create subplots for numerical features
    plt.figure(figsize=(20, len(numerical_features) * 5))
    plot_index = 1

    for col in numerical_features:
        plt.subplot(len(numerical_features), 1, plot_index)
        ax = sns.histplot(df[col], kde=False, color='blue')
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add data labels to histogram bins
        for p in ax.patches:
            height = int(p.get_height())
            if height > 0:
                ax.annotate(f'{height}', 
                            (p.get_x() + p.get_width() / 2, height), 
                            ha='center', va='bottom', fontsize=10, color='black')
        plot_index += 1

    # Adjust layout
    plt.suptitle('Distribution of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.show()
    
# =================== EDA: OUTLIER DETECTION =====================
def detect_outliers(df, target_column='HeartDisease', z_threshold=3, iqr_multiplier=1.5, plot=True):
    """
    Detect and visualize outliers in numerical features of a DataFrame.

    Parameters:
    - df: Input DataFrame.
    - target_column: Name of the target column to exclude (default is 'HeartDisease').
    - z_threshold: Threshold for Z-score method (default is 3).
    - iqr_multiplier: Multiplier for IQR method (default is 1.5).
    - plot: Whether to plot visualizations (default is True).

    Returns:
    - outliers_z: DataFrame indicating outliers detected by Z-score method.
    - outliers_iqr: DataFrame indicating outliers detected by IQR method.
    """
    # Select numerical features, excluding the target column
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_features = numerical_features[numerical_features != target_column]

    # Boxplots to visually detect outliers
    if plot:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_features, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(y=df[col], color='orange')
            sns.stripplot(y=df[col], color='blue', alpha=0.5)  # Overlay data points
            plt.title(f'Boxplot of {col}', fontsize=20, fontweight = 'bold')  # Increase title font size
            plt.xlabel(col, fontsize=12)  # Increase x-axis label font size
            plt.ylabel('Values', fontsize=12)  # Increase y-axis label font size
            plt.xticks(fontsize=12)  # Increase x-axis tick label font size
            plt.yticks(fontsize=12)  # Increase y-axis tick label font size
        #plt.suptitle('Boxplots for Outlier Detection', fontsize=16)  # Increase suptitle font size
        plt.tight_layout()
        plt.show()

    # Calculate Z-scores for numerical features
    z_scores = np.abs(zscore(df[numerical_features]))

    # Detect outliers using Z-score method
    outliers_z = (z_scores > z_threshold)

    # Print the number of outliers for each feature
    print("Number of outliers detected using Z-scores:")
    print(outliers_z.sum())

    # Outlier detection using IQR method
    Q1 = df[numerical_features].quantile(0.25)
    Q3 = df[numerical_features].quantile(0.75)
    IQR = Q3 - Q1

    # Detect outliers using IQR method
    outliers_iqr = ((df[numerical_features] < (Q1 - iqr_multiplier * IQR)) | 
                   (df[numerical_features] > (Q3 + iqr_multiplier * IQR)))

    # Print the number of outliers for each feature
    print("\nNumber of outliers detected using IQR method:")
    print(outliers_iqr.sum())

    # Optional: Visualize histograms with outliers highlighted
    if plot:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_features, 1):
            plt.subplot(2, 3, i)
            sns.histplot(df[col], kde=True, color='blue', label='Normal Data')
            sns.histplot(df[col][outliers_z[col]], color='red', label='Z-Score Outliers')
            sns.histplot(df[col][outliers_iqr[col]], color='green', label='IQR Outliers', alpha=0.5)
            plt.title(f'Histogram of {col} ', fontsize=20, fontweight = 'bold' )  # Increase title font size
            plt.xlabel(col, fontsize=12)  # Increase x-axis label font size
            plt.ylabel('Frequency', fontsize=12)  # Increase y-axis label font size
            plt.xticks(fontsize=12)  # Increase x-axis tick label font size
            plt.yticks(fontsize=12)  # Increase y-axis tick label font size
            plt.legend(fontsize=12)  # Increase legend font size
        #plt.suptitle('Histograms with Outliers Highlighted', fontsize=20)  # Increase suptitle font size
        plt.tight_layout()
        plt.show()

    return outliers_z, outliers_iqr

# ==================== EDA: HEATMAP CORRELATION ===========================
def plot_correlation_heatmap(df, target_column='HeartDisease', figsize=(10, 8), annot=True, cmap='coolwarm', title='Correlation Heatmap'):
    """
    Plot a correlation heatmap for the numerical features in a DataFrame.

    Parameters:
    - df: Input DataFrame.
    - target_column: Name of the target column (optional). If provided, correlations with the target column are highlighted.
    - figsize: Size of the heatmap figure (default is (10, 8)).
    - annot: Whether to annotate the heatmap with correlation values (default is True).
    - cmap: Color map for the heatmap (default is 'coolwarm').
    - title: Title of the heatmap (default is 'Correlation Heatmap').

    Returns:
    - correlation_matrix: The correlation matrix as a DataFrame.
    """

    # Select numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

    # Calculate the correlation matrix
    correlation_matrix = df[numerical_features].corr()

    # Plot the heatmap
    plt.figure(figsize=figsize)

    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=".2f", linewidths=0.5, linecolor='k', annot_kws={'size': 20})
    plt.title(title, fontsize = 30)
    plt.tight_layout()
    plt.show()

    # If a target column is provided, print correlations with the target
    if target_column and target_column in numerical_features:
        print(f"\nCorrelations with {target_column}:")
        print(correlation_matrix[target_column].sort_values(ascending=False))

    return correlation_matrix


# ===================== RELATIONAL GRAPHS=====================
def plot_categorical_features(df, target_column=None):
    """
    Create subplots for all categorical columns in the DataFrame with a single shared legend.

    Parameters:
    - df: Input DataFrame.
    - target_column: The name of the target column (optional). If provided, plots will compare categories by the target.
    """
    # Identify categorical columns (excluding the target column)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if target_column in categorical_columns:
        categorical_columns = categorical_columns.drop(target_column)

    # Determine subplot grid size
    num_cols = len(categorical_columns)
    num_rows = (num_cols + 2) // 3  # Create rows of 3 plots each

    plt.figure(figsize=(18, 6 * num_rows))
    handles, labels = None, None  # To store legend info

    # Generate subplots for each categorical column
    for i, cat_col in enumerate(categorical_columns, 1):
        plt.subplot(num_rows, 3, i)
        
        if target_column:
            # Group by categorical column and target column, then calculate the count
            grouped_data = df.groupby([cat_col, target_column]).size().reset_index(name='Count')

            # Create a grouped bar plot
            ax = sns.barplot(x=cat_col, y='Count', hue=target_column, data=grouped_data, palette='viridis')
            plt.title(f'{cat_col}', fontsize=20, weight = 'bold')

            # Add data labels for grouped bar plot
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2), textcoords='offset points'
                )

            # Capture legend handles and labels only once
            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()

            # Remove individual legends
            ax.legend_.remove()
        else:
            # Calculate the count of each category
            count_data = df[cat_col].value_counts().reset_index()
            count_data.columns = [cat_col, 'Count']

            # Create a bar plot
            ax = sns.barplot(x=cat_col, y='Count', data=count_data, palette='viridis')
            plt.title(f'Distribution of {cat_col}', fontsize=35, weight = 'bold')

            # Add data labels for single bar plot
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2), textcoords='offset points'
                )

        # Customize each subplot
        plt.xlabel(cat_col, fontsize=10)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(fontsize=10, rotation=0)
        plt.yticks(fontsize=10)

    # Create a single legend outside the subplots if there is a target column

    if target_column and handles:
        custom_labels = ['No Heart Disease', 'Heart Disease']
        plt.legend(
            handles,  # Use the existing handles (colors)
            custom_labels,  # Use custom labels
            title=target_column,  # Legend title
            fontsize=12,
            loc='upper center',  # Anchor point for the legend
            bbox_to_anchor=(1.5, 0.7)  # Position outside the subplot
        )  # Window-based position

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit legend
    plt.suptitle('Categorical Feature Distributions', fontsize=16, y=1.02)
    plt.show()

def plot_relational_graphs(df, target_column='HeartDisease'):
    """
    Plot relational graphs between numerical features and a binary target column.

    Parameters:
    - df: Input DataFrame.
    - target_column: Name of the binary target column (default is 'HeartDisease').
    """
    # Select numerical features excluding the target column
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_features = numerical_features[numerical_features != target_column]

    plt.figure(figsize=(15, 20))
    for i, col in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
        sns.violinplot(x=target_column, y=col, data=df, palette='viridis', split=True)
        plt.title(f'{col} ', fontsize=20, fontweight = 'bold')
        plt.xlabel('Heart Disease')
        plt.ylabel(col)
    
    plt.tight_layout()
    #plt.suptitle('Relational Graphs between Features and Heart Disease', fontsize=20, fontweight = 'bold', y=1.02)
    plt.show()

# ===================== MAINWORK FLOW=========================
if __name__ == "__main__":
    #load the data
    print("Loading data...")
    df = load_data()  # Replace with your actual data loading function
    print("Data loaded successfully!")
    print(df.head())  # Print the first few rows of the data
    
    #Preprocess the data
    print("\nPreprocessing data...")
    processed_data = preprocessing(df)  # Replace with your actual preprocessing function
    print("Data preprocessing completed!")
    print(processed_data.head())  # Print the first few rows of the processed data
    print(processed_data.columns) 

    plot_categorical_features(processed_data)
    plot_numerical_features(processed_data)

    #outlier detection
    print("\nDetecting outliers...")
    outliers_z, outliers_iqr = detect_outliers(df)  # Replace with your actual outlier detection function
    print("Outlier detection completed!")
    print("Outliers detected using Z-score method:")
    print(outliers_z.sum())
    print("\nOutliers detected using IQR method:")
    print(outliers_iqr.sum())

    # heatmap plot for correlation
    print("\nPlotting correlation heatmap...")
    plot_correlation_heatmap(processed_data)  # Replace with your actual heatmap function
    print("Correlation heatmap plotted!")

    plot_relational_graphs(processed_data)

    plot_categorical_features(processed_data, target_column='HeartDisease')






    