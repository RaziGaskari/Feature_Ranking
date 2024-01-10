# Author: 
    Razi Gaskari
# Date: 
    AUG 2023
# Project objective:
This project tries to provide feedback for selecting the influential
features for modeling based on multiple Unsupervised/Supervised methods 
for an ordinal/nominal classification problem.

Note - In this study, we are skipping the regular data cleaning and 
outlier detection steps, and we will focus on only feature selection.

# Summary:
All machine learning workflows depend on feature engineering, which 
comprises feature extraction and feature selection, which are 
fundamental building blocks of modern machine learning pipelines.

the objectives of feature selection include
    •	simplifying the model's complexity
    •	shorter training times
    •	avoiding the curse of dimensionality
    •	avoiding overfitting by enhancing generalization 

# ML Methos was considered in this study:

            Unsupervised                 ->         PCA,
                                                    Least-Squares,

            Supervised  
                        Nominal category ->         ANOVA,
                                                    Chi-Squared Statistic,
                                                    Logistic Regression,
                                                    Mutual Information Statistic,
                                                    XGBOOST,
                                                    Random Forest,                                                        
                                       
                        Ordinal category ->         Linear Regression,
                                                    Extra Tree,
                                              
# Main functions:
  feature_ranking.py :
    - feature_selection_unsupervised(df_input)
                    This function ranks the features based on input features using unsupervised methods.

                    Args:
                        df_input (pd.DataFrame): Input data for feature ranking analysis.

                    Returns:
                        pd.DataFrame: dataframe, Including the method's name and feature raking (descending).
                        the column name is features, indices are the methods, and the body is score (scale 0-1)
    
    - feature_selection_supervised (df_input,output,method,recursive,apply_encoder)

                    This function ranks the features based on input features and corresponding outputs using
                    supervised methods.   

                    Args:
                        df_input (pd.DataFrame): Input data for feature ranking analysis.
                        Output (np.ndarray): Output data for feature ranking analysis.
                        Method (str, optional): Nature of the output data. "nominal" or "ordinal". Defaults to "nominal".
                        Recursive (boolean, optional): Apply recursive techniques for selected methods—defaults to False.
                        apply_encoder (boolean, optional): Apply encoder to output data—defaults to False.

                    Returns:
                        pd.DataFrame: Including the method's name and feature raking (descending).
                        The column name is features, indices are the methods, and the body is score (scale 0-1) 

  feature_ranking_methods.py:
    The methods in this file can be expanded but should follow the same format for supervised or unsupervised structures.
    Of course, after that, the list of the methods (feature_ranking.py) needs to be updated.

    sample methods:
    - supervised - method_extratree(input_scale, output, feature_names,recursive) 
                    Bagged decision trees Extra Trees can be used to estimate the importance of features.

                    Args:
                      input_scale (np.ndarray): scaled input data for feature ranking analysis.
                      output (np.ndarray): Ouput data for feature ranking analysis.
                      feature_names (list[str]): name of the input features.
                      Recursive (boolean): Apply recursive techniques for selected methods.

                    Returns:
                      Dict[str, float]: features name and their scores (0-1)
    
    - unsupervised - method_pca(input_scale, feature_names) 
                    Principal Component Analysis (PCA) is generally called a data reduction technique. 
                    A property of PCA is that you can choose the number of dimensions or principal component
                    in the transformed result.

                    Args:
                      input_scale (np.ndarray): scaled input data for feature ranking analysis.
                      feature_names (list[str]): name of the input features.
                    Returns:
                      Dict[str, float]: features name and their scores (0-1)
  
  feature_ranking_pdf.py:
    - create_report (df,path)

                    This function creates a PDF report, including the graph and table for feature ranking.
                    Limitation: max 15 features can be shown in the report.

                    Args:
                        df (pd.DataFrame): Ranked features, the column name is features, indices are the methods, 
                        and the body is score (scale 0-1)

                        path (str): path for destination pdf file 
                    Returns None

# Step-by-step example:
    Supervised example (example_supervised.py):
        The data used for this example is about measurements of the geometrical properties of kernels
        belonging to three different wheat varieties.
        - Load the input data as a dataframe
            df = pd.read_csv(os.path.join(file_path, "seeds_dataset.csv"))
        - separate input and output data
            df_input = df.iloc[:, 0:-1]
            df_output = df.iloc[:, -1]
        - Call ranking feature function: the data is considered "nominal," and we used the recursive method
            df_result = feature_selection_supervised(df_input, df_output, "nominal", True)
        - Create a PDF report including the graph and table for feature ranking
            create_report(df_result, result_path)

    Unsupervised example (example_unsupervised.py):
        The data used for this example is about the Boston Housing Dataset. It is derived from information
        collected by the U.S. Census Service concerning housing in the area of Boston MA. 
        - Load the input data as a dataframe
            df = pd.read_csv(os.path.join(file_path, "Boston_Housing.csv"))
        - separate input data
            df_input = df.iloc[:, 0:-1]
        - Call ranking feature function
            df_result = feature_selection_unsupervised(df_input)
        - Create a PDF report including the graph and table for feature ranking
            create_report(df_result, result_path)

## Methods:

- Principal Component Analysis (PCA): Generally, this is called a data reduction technique. 
  A property of PCA is that you can choose the number of dimensions or principal component
  in the transformed result.

- Extra Tree: Bagged decision trees like Random Forest and Extra Trees can be used to estimate
 the importance of features.

- Chi-Squared Statistic: The chi-squared test is used to determine whether there is a significant
  difference between the expected frequencies or proportions or distribution and the observed 
  frequencies or proportions or distribution in one or more categories.

- Random Forests: A kind of Bagging Algorithm aggregates a specified number of decision trees.

- Least-Squares Feature Selection: It is one of the primary techniques used to correct prediction error 
  results for linear regression.

- Linear Regression: It is focused on determining the relationship between one independent variable (input)
  and one dependent variable(output).

- ANOVA is a statistical test used to analyze the difference between the means of more than two groups.

- XGBOOST: It is a supervised learning algorithm that attempts to predict a target variable accurately 
  by combining the estimates of a set of simpler, weaker models.

- Mutual Information Statistic: It is the  mutual information between two random variables measures non-linear
  relations between them.

- Logistic Regression: It estimates the probability of an event occurring based on a given dataset of 
  independent variables.

## Definitions: 

- Nominal data is sometimes called “labeled” or “named” data. Examples
  of nominal data could be a different categories's name.

- Ordinal data is a data type with a set order or scale. Example of ordinal
  data could be any regression values.

- Recursive Feature Elimination: The Recursive Feature Elimination (or RFE) works
  by recursively removing attributes and building a model on those attributes that remain.

