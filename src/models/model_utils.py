import time
import mlflow
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle as pickle
import json, logging
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

def load_labels_for_training(dataframe):
    """
    Returns the label column of the given dataframe. The label must already be discretized.
        Args: dataframe containing labels "benign" and "malicious"
        Returns: dataframe holding labels
    """
    return dataframe[[config['data']['label']]]

def load_data(dataframe):
    """ Return features only

        Args: dataframe (raw)
        
        Returns: dataframe of features only
    """

    dataframe = dataframe[list(config['data']['columns'].keys())]
    dataframe = dataframe.select_dtypes(exclude=['object']) # Drop object features
    # if 'longest_word' in dataframe.columns:
    #     dataframe.drop(columns=['longest_word'], inplace=True)
    # if 'sld' in dataframe.columns:
    #     dataframe.drop(columns=['sld'], inplace=True)
    return dataframe

def metrics_calculations(y_test, y_pred):
    """ Calculate accuracy, f score, recall and precision and prints all values
        then return values in a dictionary
        
        Args: resulting y_test and y_pred values from models
        
        Returns: data_log containing the accuracy, f score, recall, and precision
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)

    data_log = {'accuracy': accuracy,
                'f1score': f1score,
                'recall': recall,
                'precision': precision,
                'runtime': 0
                }
    return data_log


def mlflow_logger(data_array):
    '''
    Logger for mlflow which logs all data in mlflow runner for each time models are run
    To check mlflow results go to this directory and run "mlflow ui" in terminal
    Results will be shown at http://localhost:5000

    data_array will contain data in the order [accuracy, f1score, recall, precision, runtime]
    '''
    with mlflow.start_run():
        mlflow.log_metric('Accuracy', data_array['accuracy'])
        mlflow.log_metric('F Score', data_array['f1score'])
        mlflow.log_metric('Precision', data_array['precision'])
        mlflow.log_metric('Recall', data_array['recall'])
        mlflow.log_metric('Run Time', float(data_array['runtime']))


def random_forest(x, y):
    """ Runs a random forest model to produce metrics and save the model
        and use for predictions
        
        Arg: x is the set of features/parameters
        Arg: y is the set of labels/answers
    """
    logger = logging.getLogger(__name__)
    logger.info("RANDOM FOREST")

    start = time.time()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config['data']['random-forest']['test-size'])

    # Create random forest classifier
    clf = RandomForestClassifier(n_estimators=config['data']['random-forest']['n-estimators'],
                                 max_depth=config['data']['random-forest']['max-depth'],
                                 criterion=config['data']['random-forest']['criterion'],
                                 class_weight=config['data']['random-forest']['class_weight'],
                                 bootstrap=config['data']['random-forest']['bootstrap'],
                                 random_state=config['data']['random-forest']['random_state'])

    # train using the training data
    clf.fit(x_train, y_train.values.ravel())

    filename = config['data']['random-forest']['filename']

    y_pred = clf.predict(x_test)
    data_log = metrics_calculations(y_test, y_pred.round())

    # to be replaced with logger.info when logger is implemented
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix: [ [TN FP] [FN TP] ]\n", confusion_matrix(y_test, y_pred))

    finish = time.time()

    total_runtime = str(finish - start)
    logger.info(total_runtime) # For ELK stack 
    data_log.update(runtime=total_runtime)
    print(filename)
    # SAVE THE MODEL
    pickle.dump(clf, open(filename, 'wb'))

    return data_log

def decision_tree(x, y):
    """ Runs decision tree model to produce metrics and save the model
        and use for predictions
        
        Arg: x is the set of features/parameters
        Arg: y is the set of labels/answers
    """
    logger = logging.getLogger(__name__)
    logger.info("DECISION TREE")

    start = time.time()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config['data']['decision-tree']['test-size'])

    # Create decision tree classifier
    clf = DecisionTreeClassifier(max_depth=config['data']['decision-tree']['max-depth'],
                                 criterion=config['data']['decision-tree']['criterion'],
                                 class_weight=config['data']['decision-tree']['class_weight'],
                                 random_state=config['data']['decision-tree']['random_state'])

    # train using the training data
    clf.fit(x_train, y_train.values.ravel())

    filename = config['data']['decision-tree']['filename']

    y_pred = clf.predict(x_test)
    data_log = metrics_calculations(y_test, y_pred.round())

    # to be replaced with logger.info when logger is implemented
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix: [ [TN FP] [FN TP] ]\n", confusion_matrix(y_test, y_pred))

    finish = time.time()

    total_runtime = str(finish - start)
    logger.info(total_runtime) # For ELK stack 
    data_log.update(runtime=total_runtime)
    print(filename)
    # SAVE THE MODEL
    pickle.dump(clf, open(filename, 'wb'))

    return data_log

def xgboost(x, y):
    """ Runs xgboost model to produce metrics and save the model
        and use for predictions
        
        Arg: x is the set of features/parameters
        Arg: y is the set of labels/answers
    """
    logger = logging.getLogger(__name__)
    logger.info("XGBOOST")

    start = time.time()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config['data']['xgboost']['test-size'])

    # Create xgboost classifier
    clf = xgb.XGBClassifier(objective=config['data']['xgboost']['objective'],
                            random_state=config['data']['xgboost']['random_state'])

    # train using the training data
    clf.fit(x_train, y_train.values.ravel())

    filename = config['data']['xgboost']['filename']

    y_pred = clf.predict(x_test)
    data_log = metrics_calculations(y_test, y_pred.round())

    # to be replaced with logger.info when logger is implemented
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix: [ [TN FP] [FN TP] ]\n", confusion_matrix(y_test, y_pred))

    finish = time.time()

    total_runtime = str(finish - start)
    logger.info(total_runtime) # For ELK stack 
    data_log.update(runtime=total_runtime)
    print(filename)
    # SAVE THE MODEL
    pickle.dump(clf, open(filename, 'wb'))

    return data_log

def mlp(x, y):
    """ Runs Multilayer Perceptron model to produce metrics and save the model
        and use for predictions
        
        Arg: x is the set of features/parameters
        Arg: y is the set of labels/answers
    """
    logger = logging.getLogger(__name__)
    logger.info("XGBOOST")

    start = time.time()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config['data']['xgboost']['test-size'])

    # Scaling data
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    # Create mlp classifier & train using the training data
    clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),
                        activation=config['data']['mlp']['activation'],
                        random_state=config['data']['mlp']['random_state']).fit(x_train_scaled, y_train.values.ravel())

    filename = config['data']['mlp']['filename']

    y_pred = clf.predict(x_test_scaled)
    data_log = metrics_calculations(y_test, y_pred.round())

    # to be replaced with logger.info when logger is implemented
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix: [ [TN FP] [FN TP] ]\n", confusion_matrix(y_test, y_pred))

    finish = time.time()

    total_runtime = str(finish - start)
    logger.info(total_runtime) # For ELK stack 
    data_log.update(runtime=total_runtime)
    print(filename)
    # SAVE THE MODEL
    pickle.dump(clf, open(filename, 'wb'))

    return data_log
