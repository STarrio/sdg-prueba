from sre_parse import fix_flags


def model_evaluation(origin_csv_path,model_store_path, estimator, airflow_tld):

    import pandas as pd
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    from joblib import load
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Load fitted model
    # 2. Load hold-out data
    # 3. Prediction
    # 4. Get metrics
    # 5. Show metrics

    # Load fitted model pipeline
    pipe = load(model_store_path)

    # Load holdout set
    holdout_df = pd.read_csv(origin_csv_path)
    holdout_df.drop(columns='Unnamed: 0',inplace=True)

    # Separate features (X) from target (y)
    X_test, y_test = holdout_df.drop(columns='churn'), holdout_df.churn

    # Score the accuracy on the test set
    accuracy = pipe.score(X_test, y_test)

    y_pred = pipe.predict(X_test)

    # Prints the model accuracy
    print(f'{accuracy:.1%} test set accuracy')

    # Print classification report
    print(classification_report(y_test, y_pred))

    if(estimator == 'logreg'):

        weights = pd.Series(pipe.named_steps['classifier'].coef_[0],
                    index=X_test.columns.values)

        # Positive importances
        imp = weights.sort_values(ascending = False)[:10].plot(kind='bar',figsize=(25,25))
        fig = imp.get_figure()
        fig.savefig(f'{airflow_tld}/dags/results/positive_importances.png')

        # Negative importances
        imn = weights.sort_values(ascending = False)[-10:].plot(kind='bar',figsize=(25,25))
        fig = imn.get_figure()
        fig.savefig(f'{airflow_tld}/dags/results/negative_importances.png')
    else:
        pass
        features = X_test.columns
        feature_importances = pipe.named_steps['classifier'].feature_importances_
        N_FEATURES=20
        features_df = pd.DataFrame({'Features':features[:N_FEATURES], 'Importances':feature_importances[:N_FEATURES]}).sort_values('Importances', ascending=False)
        g=sns.barplot(data=features_df,x='Importances',y='Features')
        plt.savefig(f'{airflow_tld}/dags/results/feature_importances.png')


    disp = ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_test,
        y_test,
        
        cmap=plt.cm.Blues,
        normalize=None,
    )
    disp.ax_.set_title('CM')

    plt.savefig(f'{airflow_tld}/dags/results/confusion_matrix.png')