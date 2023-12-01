def model_fitting(origin_csv_path, dest_csv_path, model_store_path, estimator):

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from joblib import dump
    # 1. Load preprocessed dataset
    # 2. Separate features from target
    # 3. Fit model
    # 4. Save fitted model as pickle

    ESTIMATORS = {
        'logreg': LogisticRegression(max_iter=5000),
        'gb': GradientBoostingClassifier(n_estimators= 115, min_samples_split= 4, max_features= 1.0, max_depth= 5),
        'rf': RandomForestClassifier(max_depth=10,max_features='sqrt',min_samples_leaf=2,n_estimators=250)
        }

    # Load processed dataset CSV
    df = pd.read_csv(origin_csv_path)
    
    # Separate features (X) from target (y)
    X, y = df.drop(columns='churn'), df.churn

    # Find numerical features for standardization
    numericals = X.select_dtypes(include=[np.number]).columns.values

    # Create training set and holdout set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=616)

    # Create and save holdout set for model evaluation
    X_holdout = X_test
    X_holdout['churn'] = y_test
    X_holdout.to_csv(dest_csv_path)

    # MODEL PIPELINE
    # ColumnTransformer to apply a StandardScaler only for numerical columns
    step1 = ColumnTransformer(
        [
            ("scaler", StandardScaler(), numericals),
        ],
        remainder="passthrough",
    )

    # Estimator instance
    model = ESTIMATORS[estimator]

    # Pipeline to apply ColumnTransformer followed by a LogisticRegression estimator
    pipe = Pipeline([
        ('scaler', step1),
        ('classifier', model)])

    # Fit the pipeline to the training data
    pipe.fit(X_train, y_train)

    # Save model pipeline as a pickle
    dump(pipe, model_store_path)

  