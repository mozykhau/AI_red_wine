# Red Wine Classifier
# Dataset https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
# EE4686 Machine Learning, a Bayesian Perspective
# Assignment 3
#
# Author: Maksym Kyryliuk 5173986
# Email: m.s.kyryliuk@student.tudelft.nl
# Date: 08.07.2021
#
# Sources:
# Paulo  Cortez,  Ant ́onio  Cerdeira,  Fernando  Almeida,  et  al.  “Modeling  wine  preferences  by  data
# mining  from  physico-chemical  properties”.  In:Decision Support Systems47.4  (Nov.  2009),  pp.  547–553.ISSN:
# 01679236.DOI:  10.1016/j.dss.2009.05.016.URL: https://linkinghub.elsevier.com/retrieve/pii/S0167923609001377.
#
# Sunny Kumar, Kanika Agrawal, and Nelshan Mandan. “Red wine quality prediction using machine learning techniques”.
# In:2020 International Conference on Computer Communication and Informatics, ICCCI 2020.  2020.DOI:  10 . 1109
# /ICCCI48352.2020.9104095
#
# https://www.kaggle.com/namanmanchanda/red-wine-eda-and-classification
#
# https://www.kaggle.com/d4rklucif3r/red-wine-quality#Training-Classifiers-on-Training-Set-and-drawing-Inference
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
dataset = pd.read_csv('winequality-red.csv')

print(dataset.describe())

# # Check quality distribution
sns.set_style('whitegrid')
plt.figure(figsize=(4, 4))
sns.countplot(x="quality", data=dataset, palette='Accent')

# # Check correlation matrix

plt.figure(figsize=(8, 8))
matrix = np.triu(dataset.corr())
sns.heatmap(dataset.corr(), annot=True, linewidth=.8, mask=matrix, cmap="mako");


# plt.show()


# Remove outliers
def remove_outliers(in_dataset, fa_lim, va_lim, ca_lim, rs_lim, chl_lim, fsd_lim, tsd_lim, sul_lim, alc_lim, den_lim,
                    ph_lim):
    in_dataset = in_dataset.drop(in_dataset[in_dataset["fixed acidity"] > fa_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["volatile acidity"] > va_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["citric acid"] > ca_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["residual sugar"] > rs_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["chlorides"] > chl_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["free sulfur dioxide"] > fsd_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["total sulfur dioxide"] > tsd_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["sulphates"] > sul_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["alcohol"] > alc_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["density"] < den_lim].index)
    in_dataset = in_dataset.drop(in_dataset[in_dataset["pH"] > ph_lim].index)
    return in_dataset


# Insert limits to reduce number of outliers
dataset = remove_outliers(dataset, 15, 1.3, 0.9, 13, 0.4, 55, 200, 1.5, 14, 0.99, 3.9)
print("Final shape", dataset.shape)


def reduce_quality_classes(dataset, number):
    if number == 2:
        dataset['quality'] = np.where(dataset['quality'] > 6, 1, 0)
    elif number == 3:
        dataset['quality'] = np.where(dataset['quality'] < 4.5, 0, (np.where(dataset['quality'] > 6.5, 2, 1)))
    print('New classes count:')
    print(dataset['quality'].value_counts(), '\n')


# Reduce quality classes to 2
class_number = 6
reduce_quality_classes(dataset, class_number)


# Skewness check
def skewness_check(dataset):
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(4, 3)
    gs.update(wspace=0.5, hspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    ax10 = fig.add_subplot(gs[3, 0])
    ax11 = fig.add_subplot(gs[3, 1])

    ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax1, x=dataset['fixed acidity'], color="#3339FF", kde=True)
    Xstart, Xend = ax1.get_xlim()
    Ystart, Yend = ax1.get_ylim()
    ax1.text(Xstart, Yend + (Yend * 0.15), 'fixed acidity', fontweight='bold', fontfamily='serif')
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax2, x=dataset['volatile acidity'], color="#3339FF", kde=True)
    Xstart, Xend = ax2.get_xlim()
    Ystart, Yend = ax2.get_ylim()
    ax2.text(Xstart, Yend + (Yend * 0.15), 'volatile acidity', fontweight='bold', fontfamily='serif')
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax3, x=dataset['citric acid'], color="#3339FF", kde=True)
    Xstart, Xend = ax3.get_xlim()
    Ystart, Yend = ax3.get_ylim()
    ax3.text(Xstart, Yend + (Yend * 0.15), 'citric acid', fontweight='bold', fontfamily='serif')
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    ax4.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax4, x=dataset['residual sugar'], color="#3339FF", kde=True)
    Xstart, Xend = ax4.get_xlim()
    Ystart, Yend = ax4.get_ylim()
    ax4.text(Xstart, Yend + (Yend * 0.15), 'residual sugar', fontweight='bold', fontfamily='serif')
    ax4.set_xlabel("")
    ax4.set_ylabel("")

    ax5.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax5, x=dataset['chlorides'], color="#3339FF", kde=True)
    Xstart, Xend = ax5.get_xlim()
    Ystart, Yend = ax5.get_ylim()
    ax5.text(Xstart, Yend + (Yend * 0.15), 'chlorides', fontweight='bold', fontfamily='serif')
    ax5.set_xlabel("")
    ax5.set_ylabel("")

    ax6.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax6, x=dataset['free sulfur dioxide'], color="#3339FF", kde=True)
    Xstart, Xend = ax6.get_xlim()
    Ystart, Yend = ax6.get_ylim()
    ax6.text(Xstart, Yend + (Yend * 0.15), 'free sulfur dioxide', fontweight='bold', fontfamily='serif')
    ax6.set_xlabel("")
    ax6.set_ylabel("")

    ax7.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax7, x=dataset['total sulfur dioxide'], color="#3339FF", kde=True)
    Xstart, Xend = ax7.get_xlim()
    Ystart, Yend = ax7.get_ylim()
    ax7.text(Xstart, Yend + (Yend * 0.15), 'total sulfur dioxide', fontweight='bold', fontfamily='serif')
    ax7.set_xlabel("")
    ax7.set_ylabel("")

    ax8.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax8, x=dataset['density'], color="#3339FF", kde=True)
    Xstart, Xend = ax8.get_xlim()
    Ystart, Yend = ax8.get_ylim()
    ax8.text(Xstart, Yend + (Yend * 0.15), 'density', fontweight='bold', fontfamily='serif')
    ax8.set_xlabel("")
    ax8.set_ylabel("")

    ax9.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax9, x=dataset['pH'], color="#3339FF", kde=True)
    Xstart, Xend = ax9.get_xlim()
    Ystart, Yend = ax9.get_ylim()
    ax9.text(Xstart, Yend + (Yend * 0.15), 'pH', fontweight='bold', fontfamily='serif')
    ax9.set_xlabel("")
    ax9.set_ylabel("")

    ax10.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax10, x=dataset['sulphates'], color="#3339FF", kde=True)
    Xstart, Xend = ax10.get_xlim()
    Ystart, Yend = ax10.get_ylim()
    ax10.text(Xstart, Yend + (Yend * 0.15), 'sulphates', fontweight='bold', fontfamily='serif')
    ax10.set_xlabel("")
    ax10.set_ylabel("")

    ax11.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.histplot(ax=ax11, x=dataset['alcohol'], color="#3339FF", kde=True)
    Xstart, Xend = ax11.get_xlim()
    Ystart, Yend = ax11.get_ylim()
    ax11.text(Xstart, Yend + (Yend * 0.15), 'alcohol', fontweight='bold', fontfamily='serif')
    ax11.set_xlabel("")
    ax11.set_ylabel("")
    plt.show()


# Split data into training and testing

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
print("The shape after train/test split and scaling...")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

accuracy_scores = {}


# Predictor Function
# Define predictor based on AI model and generate metrics report
# Input:
# predictor = predictor type(e.g. knn)
# params = parameter for predictor from sklearn package
# Output:
# Trained model with classification report
def predictor(predictor, params):
    global accuracy_scores
    if predictor == 'knn':
        print('Training K-Nearest Neighbours on Training Set')
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(**params)

    elif predictor == 'nb':
        print('Training Naive Bayes Classifier on Training Set')
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB(**params)

    elif predictor == 'rfc':
        print('Training Random Forest Classifier on Training Set')
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(**params)
    else:
        print('Not correct predictor name: choose between knn, nb, rfc')
        exit()
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    print('''Prediciting Test Set Result''')
    y_pred = classifier.predict(X_test)

    result = np.concatenate((y_pred.reshape(len(y_pred), 1),
                             y_test.reshape(len(y_test), 1)), 1)

    print('''Classification Report''')
    if class_number == 2:
        print(classification_report(y_test, y_pred, target_names=['0', '1'], zero_division=1))
    elif class_number == 3:
        print(classification_report(y_test, y_pred, target_names=['0', '1', '2'], zero_division=1))
    else:
        print(classification_report(y_test, y_pred, target_names=['3', '4', '5', '6', '7', '8'], zero_division=1))

    print('''Evaluating Model Performance''')
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy = ', accuracy)
    print('Training time = ', train_time, '\n')

    print('''Applying K-Fold Cross validation''')
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(
        estimator=classifier, X=X_train, y=y_train, cv=5)
    print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
    accuracy_scores[classifier] = accuracies.mean() * 100
    print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100), '\n')


# Apply classifiers

# Naive Bayes
predictor('nb', {})

# Random Forest
predictor('rfc', {'criterion': 'gini', 'max_features': 'log2', 'n_estimators': 100, 'random_state': 0})

# KNN
predictor('knn', {'algorithm': 'auto', 'n_jobs': 1,
                  'n_neighbors': 8, 'weights': 'distance'})

# Chose the best model based on accuracy after cross validation
model_accuracies = list(accuracy_scores.values())
model_names = ['Naive Bayes', 'Random Forest', 'KNN']
max_value = max(model_accuracies)
max_index = model_accuracies.index(max_value)
print('-----------------------------------------------------------------')
print('The most accurate model is ', model_names[max_index], ' with accuracy ', '{:.2f}'.format(max_value), '%')
print('-----------------------------------------------------------------')
