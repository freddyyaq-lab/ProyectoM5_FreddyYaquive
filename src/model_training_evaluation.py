# libraries 
from ft_engineering import ft_engineering
from Cargar_datos import cargarDatos
import io
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.model_selection import(
    KFold,
    cross_val_score,
    train_test_split,
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import numpy as np
import xgboost as xgb
import pandas as pd
    
# load the data
df = cargarDatos()

# features/target
X = df.drop('Pago_atiempo',axis=1) # features
y = df['Pago_atiempo']             # target

# split the data: train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y
)

# Training

## function: summarize_classification()
def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    prec = precision_score(y_test, y_pred,pos_label=0)
    recall = recall_score(y_test, y_pred,pos_label=0)
    f1 = f1_score(y_test, y_pred,pos_label=0)
    roc = roc_auc_score(y_test, y_pred)
    cantidadNoPagoAtiempo = np.count_nonzero(y_pred == 0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc,
        "casosNoPagoAtiempo": cantidadNoPagoAtiempo
    }

## function: build_model()
def build_model(
    classifier_fn,
    data_params: dict,
    test_frac: float = 0.2,
) -> dict:
    # """
    # Function to train a classification model

    # Args:
    #     classifier_fn: classification function
    #     preprocessor (ColumnTransformer): preprocessor pipeline object
    #     data_params (dict): dictionary containing 'name_of_y_col',
    #                         'names_of_x_cols', and 'dataset'
    #     test_frac (float): fraction of data for the test, default 0.2

    # Returns:
    #     dict: dictionary with the model performance metrics on train and test

    # """

    # Extract data parameters
    name_of_y_col = data_params["name_of_y_col"]
    names_of_x_cols = data_params["names_of_x_cols"]
    dataset = data_params["dataset"]

    # Separate the feature columns and the target column
    X = dataset[names_of_x_cols]
    Y = dataset[name_of_y_col]

    # Split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_frac, random_state=42
    )

    # Use the preprocessing pipeline from ft_engineering.py
    preprocessor = ft_engineering()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)



    # Create the pipeline with preprocessing and the classification model
    classifier_pipe = Pipeline(
        steps=[("model", classifier_fn)]
    )

    # Train the classifier pipeline
    model = classifier_pipe.fit(x_train, y_train)

    # Predict the test data
    y_pred = model.predict(x_test)

    # Predict the train data
    y_pred_train = model.predict(x_train)

    # Calculate the performance metrics
    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)

    kfold = KFold(n_splits=10)
    model_pipe = Pipeline(steps=[("model", model)])

    cv_results = {}
    train_results = {}

    # Ejecutamos validación cruzada
    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in scoring_metrics[:-1]:  
        cv_results[metric] = cross_val_score(
            model_pipe, x_train, y_train, cv=kfold, scoring=metric
        )
        # Se evalúa sobre el Dataset de pruebas
        model_pipe.fit(x_train, y_train)
        train_results[metric] = model_pipe.score(x_train, y_train)

    # Se convierten los resultados en un df
    cv_results_df = pd.DataFrame(cv_results)



    return {"train": train_summary, "test": test_summary}

## models to train

models = {
    "logistic": LogisticRegression(solver="liblinear",class_weight='balanced'),
    # class_weight='balanced' para manejar clases desbalanceadas
    "svc": LinearSVC(C=1.0, max_iter=1000, tol=1e-3, dual=False,class_weight='balanced'),
    "decision_tree": DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,class_weight='balanced'),
    "random_forest": RandomForestClassifier(class_weight='balanced',n_estimators=150,max_depth=7,min_samples_leaf=5,max_features='sqrt'),
    "xgboost": XGBClassifier(eval_metric='logloss',scale_pos_weight=491 / 10090,n_estimators=200,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8)
    
}

data_params = {
    "name_of_y_col": y.name,
    "names_of_x_cols": X.columns,
    "dataset": df
}

result_dict = {}

for model_name, model in models.items():
    result_dict[model_name] = build_model(model, data_params)

# results
records = []

for model_name, splits in result_dict.items():
    for split_name, metrics in splits.items():
        record = {
            "Model": model_name,
            "Data": split_name
        }
        record.update(metrics)
        records.append(record)

results_df = pd.DataFrame(records)
results_df = results_df.sort_values(
    by=["Data", "roc_auc"], ascending=[True, False]
)

print(results_df.round(4))



# for model_name, model_results in result_dict.items():
#     for data_set, metrics in model_results.items():
#         for metric_name, score in metrics.items():
#             records.append({
#                 "Model": model_name,
#                 "Data Set": data_set,
#                 "Metric": metric_name,
#                 "Score": score
#             })

# results_df = pd.DataFrame(records)

# results_df = results_df.sort_values(
#     by=["Data Set", "Score"], ascending=[True, False]
# )
# print(results_df.round(4))

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


# Evaluation

def evaluation():
    
    #results_df = pd.DataFrame({'Model': {48: 'xgboost', 49: 'xgboost', 50: 'xgboost', 51: 'xgboost', 52: 'xgboost', 53: 'xgboost', 54: 'xgboost', 55: 'xgboost', 56: 'xgboost', 57: 'xgboost', 58: 'xgboost', 59: 'xgboost'}, 'Data Set': {48: 'train', 49: 'train', 50: 'train', 51: 'train', 52: 'train', 53: 'train', 54: 'test', 55: 'test', 56: 'test', 57: 'test', 58: 'test', 59: 'test'}, 'Metric': {48: 'accuracy', 49: 'precision', 50: 'recall', 51: 'f1_score', 52: 'roc_auc', 53: 'casosNoPagoAtiempo', 54: 'accuracy', 55: 'precision', 56: 'recall', 57: 'f1_score', 58: 'roc_auc', 59: 'casosNoPagoAtiempo'}, 'Score': {48: 0.9585302457466919, 49: 0.5125, 50: 1.0, 51: 0.6776859504132231, 52: 0.9783199505867819, 53: 720.0, 54: 0.8889938592347661, 55: 0.17341040462427745, 56: 0.2459016393442623, 57: 0.2033898305084746, 58: 0.5871112206746374, 59: 173.0}})
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc","casosNoPagoAtiempo"]
    model = xgb.Booster()
    model.load_model("xgb_model.json")
    # Es una buena práctica guardar los nombres de las features para asegurar el orden
    model_features = model.feature_names
    df = ft_engineering()
    df.columns = [x.replace('__','_') for x in df.columns]
    X = df[model_features]
    X = xgb.DMatrix(X)
    Y = df["Pago_atiempo"]
    y_pred = model.predict(X)
    threshold = 0.5
    y_pred= [1 if prob >= threshold else 0 for prob in y_pred]
    # # Crear una figura con una cuadrícula de subplots (3 filas, 2 columnas)
    # fig, axes = plt.subplots(3, 2, figsize=(30, 18))
    # axes = axes.flatten() # Aplanar la matriz de ejes para iterar fácilmente
    # fig.suptitle('Evaluaciones de modelo XGBoost', fontsize=30)
    # # Iterar sobre cada métrica y crear un gráfico para ella
    # custom_titles = {
    #     'precision': 'Precisión no pago a tiempo',
    #     'recall': 'Recall no pago a tiempo',
    #     'f1_score': 'F1 no pago a tiempo',
    #     'accuracy': 'Accuracy General',
    #     'roc_auc': 'ROC AUC',
    #     'casosNoPagoAtiempo': 'Conteo Casos No Pago a Tiempo'
    # }
    # for i, metric in enumerate(metrics_to_plot):
    #     ax = axes[i]
    #     # Filtrar el DataFrame para la métrica actual
    #     metric_df = results_df[results_df["Metric"] == metric]
        
    #     # Crear el gráfico de barras agrupado
    #     sns.barplot(data=metric_df, x="Model", y="Score", hue="Data Set", ax=ax, palette="cividis")
    #     ax.legend(fontsize=18)
    #     title = custom_titles.get(metric, metric.replace("_", " ").title())
    #     ax.set_title(title, fontsize=24)
    #     ax.set_xticks([])
    #     ax.set_ylabel("Puntuación", fontsize=18)
    #     ax.set_xlabel("")
    #     ax.tick_params(axis='x', rotation=45, labelsize=18)
    #     ax.tick_params(axis='y', labelsize=18)
    #     # Ajustar el límite del eje Y para que la comparación sea justa
    #     # Para ROC AUC, la escala es de 0.45 a 1.05 para ver mejor las diferencias
    #     if metric == 'roc_auc':
    #         ax.set_ylim(0.45, 1.05)
    #     elif metric == 'casosNoPagoAtiempo':
    #         max_no_pago = results_df[results_df["Metric"] == "casosNoPagoAtiempo"]["Score"].max()
    #         ax.set_ylim(0, max_no_pago + 5)
    #     else:
    #         ax.set_ylim(0, 1.05)

    # # Ocultar el último subplot que no se usa
    # if len(metrics_to_plot) < len(axes):
    #     axes[-1].set_visible(False)

    # # Ajustar el diseño para que no se superpongan los títulos
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # plt.close(fig)
    buf.seek(0)
    
    return buf