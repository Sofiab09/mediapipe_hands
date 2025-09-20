import pandas as pd
import joblib
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main(args):
    df = pd.read_csv(args.input)
    if "label" not in df.columns:
        raise ValueError("El CSV debe tener una columna 'label' al final.")
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # codificar etiquetas
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # estandarizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Entrena un RandomForest simple
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Guardar modelo, label encoder y scaler
    joblib.dump(clf, args.model)
    joblib.dump(le, args.labels)
    joblib.dump(scaler, args.scaler)
    print(f"Guardado modelo en {args.model}, etiquetas en {args.labels}, scaler en {args.scaler}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="gestures.csv", help="CSV con coordenadas y label")
    parser.add_argument("--model", default="modelo.pkl", help="Ruta para guardar el modelo")
    parser.add_argument("--labels", default="labels.pkl", help="Ruta para guardar LabelEncoder")
    parser.add_argument("--scaler", default="scaler.pkl", help="Ruta para guardar StandardScaler")
    args = parser.parse_args()
    main(args)
