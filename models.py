import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import os
import sys
from datetime import datetime

# Suprimir TODOS los warnings de manera más agresiva
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suprimir específicamente ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# Configuración de validación cruzada
FOLDS = 10  # Número de folds para validación cruzada
SCREEN_WIDTH = 100
WIDTH_BETWEEN_DATASETS = SCREEN_WIDTH + 50

class OutputLogger:
    """Clase para capturar la salida y guardarla en un archivo"""
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_results_{timestamp}.txt"
        self.filename = filename
        self.terminal = sys.stdout
        self.log_file = None
        
    def start_logging(self):
        """Inicia el logging de la salida"""
        self.log_file = open(self.filename, 'w', encoding='utf-8')
        sys.stdout = self
        
    def stop_logging(self):
        """Detiene el logging y restaura la salida normal"""
        if self.log_file:
            self.log_file.close()
            sys.stdout = self.terminal
            print(f"\nResultados guardados en: {self.filename}")
    
    def write(self, message):
        """Escribe tanto en terminal como en archivo"""
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        """Fuerza la escritura del buffer"""
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()


class MLModelTrainer:
    def __init__(self, data, target_name):
        """
        Inicializa el entrenador de modelos ML
        
        Args:
            data (pd.DataFrame): DataFrame con las características y el target
            target_name (str): nombre de la columna objetivo
        """
        self.data = data
        self.X = data.drop(target_name, axis=1)
        self.y = data[target_name]
        self.models = {
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(),
            "LogisticRegression": LogisticRegression(max_iter=2000)
        }
        
        # Parámetros para GridSearch
        self.param_grids = {
            "RandomForest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            "SVM": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'degree': [2, 3, 4]
            },
            "LogisticRegression": {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 5000]
            }
        }

    def train_and_evaluate(self):
        """Entrena y evalúa todos los modelos"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        results = {}
        for name, model in self.models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            print(f"\n=== {name} ===")
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred))
        return results

    def cross_validate_models(self, cv_folds=FOLDS, use_stratified=True):
        """
        Realiza validación cruzada con k-fold para todos los modelos
        
        Args:
            cv_folds (int): Número de folds para la validación cruzada
            use_stratified (bool): Si usar StratifiedKFold para mantener proporción de clases
        
        Returns:
            dict: Resultados de validación cruzada para cada modelo
        """
        # Preparar los datos
        X_scaled = StandardScaler().fit_transform(self.X)
        
        # Seleccionar el tipo de fold
        if use_stratified:
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        
        print(f"\n{'='*50}")
        print(f"VALIDACIÓN CRUZADA CON {cv_folds} FOLDS")
        print(f"{'='*50}")
        
        for name, model in self.models.items():
            print(f"\n--- Evaluando {name} ---")
            
            # Realizar validación cruzada con supresión de warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    model, X_scaled, self.y, 
                    cv=kfold, scoring='accuracy', n_jobs=-1
                )
            
            # Calcular estadísticas
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            min_score = cv_scores.min()
            max_score = cv_scores.max()
            
            cv_results[name] = {
                'scores': cv_scores,
                'mean': mean_score,
                'std': std_score,
                'min': min_score,
                'max': max_score
            }
            
            print(f"Puntuaciones por fold: {cv_scores}")
            print(f"Media: {mean_score:.4f} (+/- {std_score * 2:.4f})")
            print(f"Rango: [{min_score:.4f}, {max_score:.4f}]")
        
        return cv_results

    def grid_search_models(self, cv_folds=FOLDS, use_stratified=True):
        """
        Realiza GridSearch con validación cruzada para optimizar hiperparámetros
        
        Args:
            cv_folds (int): Número de folds para la validación cruzada
            use_stratified (bool): Si usar StratifiedKFold para mantener proporción de clases
        
        Returns:
            dict: Mejores modelos y resultados de GridSearch
        """
        # Preparar los datos
        X_scaled = StandardScaler().fit_transform(self.X)
        
        # Seleccionar el tipo de fold
        if use_stratified:
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        grid_results = {}
        
        print(f"\n{'='*SCREEN_WIDTH}")
        print(f"GRID SEARCH CON {cv_folds} FOLDS".center(SCREEN_WIDTH))
        print(f"{'='*SCREEN_WIDTH}")
        
        for name, model in self.models.items():
            print(f"\n--- Optimizando {name} ---")
            
            # Crear GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grids[name],
                cv=kfold,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Realizar GridSearch con supresión de warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_search.fit(X_scaled, self.y)
            
            # Guardar resultados
            grid_results[name] = {
                'best_model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score: {grid_search.best_score_:.4f}")
            
            # Calcular mejora de manera segura (manejar nan)
            mean_scores = grid_search.cv_results_['mean_test_score']
            # Filtrar valores nan y calcular media
            valid_scores = mean_scores[~np.isnan(mean_scores)]
            if len(valid_scores) > 0:
                mean_valid_score = valid_scores.mean()
                improvement = grid_search.best_score_ - mean_valid_score
                print(f"Mejora sobre modelo base: {improvement:.4f}")
            else:
                print("Mejora sobre modelo base: No calculable (todos los scores son nan)")
        
        return grid_results

    def compare_evaluation_methods(self, cv_folds=FOLDS):
        """
        Compara la evaluación simple vs validación cruzada
        
        Args:
            cv_folds (int): Número de folds para la validación cruzada
        """
        print(f"\n{'='*60}")
        print("COMPARACIÓN: EVALUACIÓN SIMPLE vs VALIDACIÓN CRUZADA")
        print(f"{'='*60}")
        
        # Evaluación simple
        print("\n1. EVALUACIÓN SIMPLE (Train/Test Split):")
        simple_results = self.train_and_evaluate()
        
        # Validación cruzada
        print(f"\n2. VALIDACIÓN CRUZADA ({cv_folds} folds):")
        cv_results = self.cross_validate_models(cv_folds)
        
        # Comparación
        print(f"\n{'='*60}")
        print("RESUMEN COMPARATIVO:")
        print(f"{'='*60}")
        print(f"{'Modelo':<20} {'Simple':<10} {'CV Media':<10} {'CV Std':<10} {'Diferencia':<10}")
        print("-" * 60)
        
        for name in self.models.keys():
            simple_acc = simple_results[name]
            cv_mean = cv_results[name]['mean']
            cv_std = cv_results[name]['std']
            diff = abs(simple_acc - cv_mean)
            
            print(f"{name:<20} {simple_acc:<10.4f} {cv_mean:<10.4f} {cv_std:<10.4f} {diff:<10.4f}")
        
        return simple_results, cv_results

    def compare_base_vs_optimized(self, cv_folds=FOLDS):
        """
        Compara modelos base vs modelos optimizados con GridSearch
        
        Args:
            cv_folds (int): Número de folds para la validación cruzada
        """
        print(f"\n{'='*WIDTH_BETWEEN_DATASETS}")
        print("COMPARACIÓN: MODELOS BASE vs OPTIMIZADOS CON GRIDSEARCH".center(SCREEN_WIDTH))
        print(f"{'='*WIDTH_BETWEEN_DATASETS}")
        
        # Evaluación de modelos base
        print(f"\n1. MODELOS BASE (parámetros por defecto):")
        base_cv_results = self.cross_validate_models(cv_folds)
        
        # GridSearch para modelos optimizados
        print(f"\n2. MODELOS OPTIMIZADOS (GridSearch):")
        grid_results = self.grid_search_models(cv_folds)
        
        # Comparación
        print(f"\n{'='*WIDTH_BETWEEN_DATASETS}")
        print("RESUMEN COMPARATIVO:".center(WIDTH_BETWEEN_DATASETS))
        print(f"{'='*WIDTH_BETWEEN_DATASETS}")
        print(f"{'Modelo':<20} {'Base CV':<12} {'Optimizado':<12} {'Mejora':<12} {'Mejora %':<12}")
        print("-" * 80)
        
        for name in self.models.keys():
            base_mean = base_cv_results[name]['mean']
            optimized_score = grid_results[name]['best_score']
            improvement = optimized_score - base_mean
            
            # Calcular porcentaje de mejora de manera segura
            if base_mean != 0 and not np.isnan(base_mean):
                improvement_pct = (improvement / base_mean) * 100
            else:
                improvement_pct = 0.0
            
            print(f"{name:<20} {base_mean:<12.4f} {optimized_score:<12.4f} {improvement:<12.4f} {improvement_pct:<12.2f}%")
        
        return base_cv_results, grid_results


def run_iris():
    iris = load_iris(as_frame=True)
    df = iris.frame
    trainer = MLModelTrainer(df, "target")
    print()
    print("#"*WIDTH_BETWEEN_DATASETS)
    print("IRIS DATASET".center(WIDTH_BETWEEN_DATASETS))
    print("#"*WIDTH_BETWEEN_DATASETS)
    print()
    # Comparar métodos de evaluación
    simple_results, cv_results = trainer.compare_evaluation_methods()
    
    # Comparar modelos base vs optimizados
    base_cv_results, grid_results = trainer.compare_base_vs_optimized()
    
    return simple_results, cv_results, base_cv_results, grid_results


def run_wine():
    wine = load_wine(as_frame=True)
    df = wine.frame
    trainer = MLModelTrainer(df, "target")
    print()
    print("#"*WIDTH_BETWEEN_DATASETS)
    print("WINE DATASET".center(WIDTH_BETWEEN_DATASETS))
    print("#"*WIDTH_BETWEEN_DATASETS)
    print()
    # Comparar métodos de evaluación
    simple_results, cv_results = trainer.compare_evaluation_methods()
    
    # Comparar modelos base vs optimizados
    base_cv_results, grid_results = trainer.compare_base_vs_optimized()
    
    return simple_results, cv_results, base_cv_results, grid_results


def run_breast_cancer():
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    trainer = MLModelTrainer(df, "target")
    print()
    print("#"*WIDTH_BETWEEN_DATASETS)
    print("BREAST CANCER DATASET".center(WIDTH_BETWEEN_DATASETS)) 
    print("#"*WIDTH_BETWEEN_DATASETS)
    print()
    
    # Comparar métodos de evaluación
    simple_results, cv_results = trainer.compare_evaluation_methods()
    
    # Comparar modelos base vs optimizados
    base_cv_results, grid_results = trainer.compare_base_vs_optimized()
    
    return simple_results, cv_results, base_cv_results, grid_results


def run_custom_validation(dataset_name, data, target_name, cv_folds=FOLDS):
    """
    Función genérica para ejecutar validación cruzada en cualquier dataset
    
    Args:
        dataset_name (str): Nombre del dataset
        data (pd.DataFrame): DataFrame con los datos
        target_name (str): Nombre de la columna objetivo
        cv_folds (int): Número de folds para validación cruzada
    """
    trainer = MLModelTrainer(data, target_name)
    print()
    print("#"*WIDTH_BETWEEN_DATASETS)
    print(f"{dataset_name.upper()} DATASET".center(WIDTH_BETWEEN_DATASETS))
    print("#"*WIDTH_BETWEEN_DATASETS)
    print()
    
    # Comparar métodos de evaluación
    simple_results, cv_results = trainer.compare_evaluation_methods(cv_folds=cv_folds)
    
    # Comparar modelos base vs optimizados
    base_cv_results, grid_results = trainer.compare_base_vs_optimized(cv_folds)
    
    return simple_results, cv_results, base_cv_results, grid_results


def run_grid_search_only(dataset_name, data, target_name, cv_folds=FOLDS):
    """
    Función para ejecutar solo GridSearch sin comparaciones
    
    Args:
        dataset_name (str): Nombre del dataset
        data (pd.DataFrame): DataFrame con los datos
        target_name (str): Nombre de la columna objetivo
        cv_folds (int): Número de folds para validación cruzada
    """
    trainer = MLModelTrainer(data, target_name)
    print()
    print("#"*SCREEN_WIDTH)
    print(f"GRID SEARCH - {dataset_name.upper()} DATASET".center(SCREEN_WIDTH))
    print("#"*SCREEN_WIDTH)
    print()
    
    # Solo GridSearch
    grid_results = trainer.grid_search_models(cv_folds)
    
    return grid_results


def run_with_logging(filename=None, save_to_file=True):
    """
    Ejecuta el análisis completo con opción de guardar en archivo
    
    Args:
        filename (str): Nombre del archivo de salida (opcional)
        save_to_file (bool): Si guardar la salida en archivo
    """
    if save_to_file:
        logger = OutputLogger(filename)
        logger.start_logging()
    
    try:
        # Ejecutar validación cruzada en todos los datasets
        print(f"Ejecutando validación cruzada con {FOLDS} folds en todos los datasets...".center(SCREEN_WIDTH))
        print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*WIDTH_BETWEEN_DATASETS)
        print()
        
        run_iris()
        run_wine()
        run_breast_cancer()
        
        print("\n" + "="*WIDTH_BETWEEN_DATASETS)
        print("¡ANÁLISIS COMPLETADO EXITOSAMENTE!".center(WIDTH_BETWEEN_DATASETS))
        print("="*WIDTH_BETWEEN_DATASETS)
        
    except Exception as e:
        print(f"\n Error durante la ejecución: {e}")
        print("Verifica que todas las dependencias estén instaladas correctamente.")
    
    finally:
        if save_to_file:
            logger.stop_logging()


if __name__ == "__main__":
    # Ejecutar con logging automático
    run_with_logging()
