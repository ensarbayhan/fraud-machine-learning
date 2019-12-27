import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


def print_data_distribution(df):
    print("Class as pie chart:")
    print('Legitimate', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% of the dataset')
    print('Frauds', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% of the dataset')
    fig, ax = plt.subplots(1, 1)
    ax.pie(df.Class.value_counts(), autopct='%1.1f%%', labels=['Legitimate', 'Fraud'], colors=['yellowgreen', 'r'])
    plt.axis('equal')
    plt.ylabel('')


def print_data(df):
    print("shape")
    print(df.shape)
    print("")

    print("First 5 lines:")
    print(df.head(5))
    print("")

    print("describe: ")
    print(df.describe())
    print("")

    print("info: ")
    print(df.info())
    print("")

    print("distribution:")
    print_data_distribution(df)


def print_data_correlations(df):
    gs = gridspec.GridSpec(28, 1)
    plt.figure(figsize=(6, 28 * 4))
    for i, col in enumerate(df[df.iloc[:, 0:28].columns]):
        ax5 = plt.subplot(gs[i])
        sns.distplot(df[col][df.Class == 1], bins=50, color='r')
        sns.distplot(df[col][df.Class == 0], bins=50, color='g')
        ax5.set_xlabel('')
        ax5.set_title('feature: ' + str(col))
    plt.show()


def split_data(df):
    print(df.columns)
    y = df['Class'].values
    X = df.drop(['Class'], axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("train-set size: ", len(y_train), "\ntest-set size: ", len(y_test))
    print("fraud cases in test-set: ", sum(y_test))
    return X_train, X_test, y_train, y_test


def get_predictions(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    train_pred = clf.predict(X_train)
    print('train-set confusion matrix:\n', confusion_matrix(y_train, train_pred))
    return y_pred, y_pred_prob


def print_scores(key, y_test, y_pred, y_pred_prob, color):
    print('test-set confusion matrix:\n', confusion_matrix(y_test, y_pred))

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
    print("recall score: ", recall)
    print("precision score: ", precision)
    print("f1 score: ", f1)
    print("accuracy score: ", accuracy)
    print("ROC AUC: {}".format(roc_auc))
    p, r, _ = precision_recall_curve(y_test, y_pred_prob[:, 1])
    tpr, fpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
    ax1.plot(r, p, c=color, label=key)
    ax2.plot(tpr, fpr, c=color, label=key)
    return round(recall, 4), round(precision, 4), round(f1, 4), round(accuracy, 4), round(roc_auc, 4)


def scale_data(df):
    rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df


def under_sampling(df):
    df = df.sample(frac=1)
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:492]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    return normal_distributed_df.sample(frac=1, random_state=42)


def drop_features(df, drop_list):
    return df.drop(drop_list, axis=1)


def predict(label, classifier, df, color):
    print(label)
    X_train, X_test, y_train, y_test = split_data(df)
    y_pred, y_pred_prob = get_predictions(classifier, X_train, y_train, X_test)
    recall, precision, f1, accuracy, roc_auc = print_scores(label, y_test, y_pred, y_pred_prob, color)
    return recall, precision, f1, accuracy, roc_auc


# Main

# Load Data
df = pd.read_csv("creditcard.csv")
print_data(df)
print_data_correlations(df)

drop_list = ['V28', 'V26', 'V25', 'V23', 'V22', 'V20', 'V15', 'V13', 'V8']
under_sampled_df = under_sampling(df.copy())
scaled_df = scale_data(under_sampled_df.copy())
dropped_df = drop_features(scaled_df.copy(), drop_list)

colors = {
    "Naive Bayes": 'b',
    "Logisitic Regression": 'r',
    "Decision Tree": 'g',
    "KNearest": 'm'
}

classifiers = {
    "Naive Bayes": GaussianNB(),
    "Logisitic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNearest": KNeighborsClassifier()
}

steps = {
    " ": df,
    " After Undersampling": under_sampled_df,
    " After Scale Amount": scaled_df,
    " After Drop Features": dropped_df,
}

for label, for_df in steps.items():
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('PR Curve')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')

    # fig2, ax3 = plt.subplots()
    fig2 = plt.figure(figsize=(8, 8), dpi=80)
    ax3 = fig2.add_subplot(1, 1, 1)

    # hide axes
    fig2.patch.set_visible(False)
    ax3.axis('off')
    ax3.axis('tight')

    result_array = []
    for key, classifier in classifiers.items():
        print("")
        recall, precision, f1, accuracy, roc_auc = predict(key + label, classifier, for_df, colors[key])
        result_array.append([key, recall, precision, f1, accuracy, roc_auc])
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower left')
    result_df = pd.DataFrame(result_array,
                             columns=['Algorithm', 'Recall', 'Precision', 'F1_score', 'Accuracy', 'ROC AUC'])
    table = ax3.table(cellText=result_df.values, colLabels=result_df.columns, loc='center')
    table.set_fontsize(13)
    fig2.tight_layout()
    plt.show()
