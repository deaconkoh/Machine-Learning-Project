# Phishing Email Project

## Project Overview
The Phishing Email Project aims to develop and evaluate machine learning models, including a neural network, to classify emails as either "phishing" or "legitimate." Given the prevalence of phishing attacks, this project provides a practical tool for distinguishing between safe and potentially harmful emails using data-driven methods.

Project team: This project was completed collaboratively by a group of SIT (Singapore Institute of Technology) students.

## Methodology

1. **Data Collection and Exploration**  
   A labeled dataset containing both legitimate (ham) and phishing (spam) emails was obtained from Kaggle, ensuring a balanced dataset for training. Initial data exploration included:  
   - **Visualization**: A pie chart highlighted the class imbalance between ham and spam emails, with ham emails comprising 79% of the dataset.
   - **Word Clouds**: Frequently used words in spam emails (e.g., "offer," "save") versus legitimate emails (e.g., "schedule," "attach") provided insights into common phishing terms.

2. **Data Preprocessing**  
   Steps to clean and prepare the data:  
   - **Stopwords Removal**: Removed stopwords using NLTK’s default list, supplemented with additional common email terms.
   - **URL Replacement**: Replaced links with 'URL' tokens to highlight frequency rather than specific content.
   - **Punctuation Removal**: Eliminated punctuation to reduce noise.
   - **Stemming**: Applied stemming to focus on root words.
   - **Tokenization**: Converted email text into individual tokens.
   - **Vectorization**: Used TF-IDF and Word2Vec for converting text into numerical format.

3. **Feature Engineering**  
   Key feature engineering techniques include:  
   - **TF-IDF**: Highlights important words while downplaying common ones, helping focus on unique phishing terms.
   - **Word2Vec**: Maps semantically similar words closer in vector space.
   - **N-Grams**: Analyzes word sequences, such as bigrams, to capture phishing patterns (e.g., "verify account").

4. **Model Development**  
   The project explored both traditional and neural network models:  
   - **Logistic Regression**: Provided a linear baseline.
   - **Random Forest**: Ensemble method capturing feature importance effectively.
   - **Naive Bayes**: Effective with discrete data, suited to text analysis.
   - **k-Nearest Neighbors (k-NN)**: Non-parametric method useful for small datasets.
   - **XGBoost**: Ensemble gradient boosting, optimized with GridSearch for best hyperparameters.

   **Neural Network Model**  
   A hybrid CNN-RNN neural network was created to capture both local word patterns and long-range dependencies. The model integrates:  
   - **CNN Layers**: Extract local patterns like common phrases.
   - **RNN Layers**: Capture the sequence and flow of words for context.  
   
   The CNN-RNN model achieved an average accuracy of 87.08% over ten fold which is lower than the standard machine learninng algorithm. This could be largely because of our smaller dataset.

5. **Model Evaluation**  
   Models were evaluated using:  
   - **Accuracy**: Overall correct classifications.
   - **Precision, Recall, and F1 Score**: Balanced metrics to evaluate phishing detection and minimize false positives.
   - **Cross-validation**: Validated generalization across data subsets. By repeating n times, using a different fold as the test set each time, and the results are averaged.

6. **Data Visualisation**
   The results were represented using:
   - **Precision-Recall Curve**: Illustrated the trade-off between precision and recall across different classification thresholds, this is useful for evaluating models in cases with imbalanced data by showing the balance between capturing true positives and minimizing false positives.
   - **Bar Graph for Accuracy Comparison**: Compared accuracy across all models to identify which performed best in classifying emails. This bar graph is a straightforward view of overall accuracy scores for each machine learning algorithm.
   - **F1 Distribution Score**: Showed F1 scores across multiple cross-validation folds for each model, examining each model's consistency and reliability in achieving balanced precision and recall scores
   - **Feature Importance Graph**: Displayed the most influential features (e.g., specific words or patterns) that contributed to classification accuracy.


## Key Findings
- **Feature Engineer**: All the models performed better with Word2Vec as the feature engineer method, except for Logistic Regression, which showed a decrease in accuracy.
- **Hybrid Model Performance**: Traditional machine learning models outperform the hybrid neural network in terms of accuracy, which is likely due to the smaller dataset used
-  **Traditional Machine Learning Model**: XGBoost and Random Forest achieved the highest evaluation score among other models, this is liekly due to their ensemble methods, ability to focus on important features, effective handling of high-dimensional data, flexibility in tuning, and robustness to imbalanced data

## Future Work
Future directions include:
- Expanded Datasets: More recent phishing examples to account for evolving techniques.
- Advanced NLP: Implementing word embeddings for richer feature representation, capturing contextual word relationships to improve the model’s accuracy in identifying subtle phishing cues.
- Adaptive Learning: Enabling real-time updates to continuously adapt to new phishing tactics, making the model more resilient to novel attack patterns and maintaining high detection accuracy over time.
