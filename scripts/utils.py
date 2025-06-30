import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV,  SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import TargetEncoder, LabelEncoder, PowerTransformer, StandardScaler, label_binarize
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, roc_auc_score, make_scorer)
from category_encoders import TargetEncoder
from copy import deepcopy

# Cek Missing Value
def cek_missing(df):
    return pd.DataFrame({"Jumlah": df.isna().sum(), "Persentase": df.isna().mean() * 100})

# Plot Mutual Information Scores
def plot_mutual_info_scores(preprocessor, X_train, y_train, figsize=(15, 10), 
                           palette='viridis', top_n=None, random_state=42):
    
    if not hasattr(preprocessor, 'fit_transform'):
        preprocessor.fit(X_train, y_train)
    X_transformed = preprocessor.transform(X_train)

    if hasattr(X_transformed, 'toarray'):
        X_transformed = X_transformed.toarray()

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    
    mi_scores = mutual_info_classif(X_transformed, y_train, random_state=random_state)
    mi_df = pd.DataFrame({
        'Feature': feature_names,
        'MI Score': mi_scores
    }).sort_values(by='MI Score', ascending=False)
    
    if top_n is not None:
        mi_df = mi_df.head(top_n)
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='MI Score', y='Feature', data=mi_df, palette=palette)
    ax.set_title('Feature Importance by Mutual Information', pad=20, fontsize=16)
    ax.set_xlabel('Mutual Information Score', labelpad=10)
    ax.set_ylabel('Feature Name', labelpad=10)
    
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.005, p.get_y() + p.get_height()/2., 
                f'{width:.3f}', 
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return mi_df

# Target Feature Encoder
class TargetFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, to_encode):
        self.to_encode = to_encode
        self.encoder = TargetEncoder()

    def fit(self, X, y):
        X_ = X.copy().reset_index(drop=True)
        self.encoder.fit(X_[self.to_encode], y)
        return self

    def transform(self, X):
        X_ = X.copy().reset_index(drop=True)
        X_target = self.encoder.transform(X_[self.to_encode])

        target_df = pd.DataFrame(
            X_target,
            columns=self.to_encode
        )

        return pd.concat([X_.drop(columns=self.to_encode), target_df], axis=1)
    
    def get_feature_names_out(self, input_features=None):
        # Menjaga nama kolom asli untuk fitur yang diencode
        if input_features is None:
            return self.to_encode
        # Filter hanya kolom yang diencode
        return [col for col in input_features if col in self.to_encode]

# Yeo-Johnson Transformer (tanpa suffix _yeo)
class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.yeo = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        self.yeo.fit(X[self.columns])
        return self

    def transform(self, X):
        X_reset = X.reset_index(drop=True)
        X_yeo = self.yeo.transform(X_reset[self.columns])

        yeo_df = pd.DataFrame(
            X_yeo,
            columns=self.columns,  # Tidak pakai suffix _yeo
            index=X_reset.index
        )

        return pd.concat([X_reset.drop(columns=self.columns), yeo_df], axis=1)
    
    def get_feature_names_out(self, input_features=None):
        # Kembalikan nama asli
        if input_features is None:
            return self.columns
        return input_features

# Standard Scaler Transformer (tanpa suffix _scaled)
class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_reset = X.reset_index(drop=True)
        X_scaled = self.scaler.transform(X_reset[self.columns])

        scaled_df = pd.DataFrame(
            X_scaled,
            columns=self.columns,  # Tidak pakai suffix _scaled
            index=X_reset.index
        )

        return pd.concat([X_reset.drop(columns=self.columns), scaled_df], axis=1)
    
    def get_feature_names_out(self, input_features=None):
        # Kembalikan nama asli
        if input_features is None:
            return self.columns
        return input_features

# Stratified K-Fold for Multiclass Classification
class StratifiedKFoldMulticlass(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_folds=5, random_state=None, method='predict_proba'):
        self.estimator = estimator
        self.n_folds = n_folds
        self.random_state = random_state
        self.method = method
        self.fold_models_ = []
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        if type_of_target(y) not in ['multiclass']:
            raise ValueError("Target harus multiclass!")
        
        self.classes_ = np.unique(y)
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        self.fold_models_ = []
        for train_idx, _ in skf.split(X, y):
            model = deepcopy(self.estimator)
            if sample_weight is not None:
                model.fit(X[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
            else:
                model.fit(X[train_idx], y[train_idx])
            self.fold_models_.append(model)
        return self

    def transform(self, X):
        if self.method == 'predict_proba':
            probas = np.zeros((len(X), len(self.classes_)))
            for model in self.fold_models_:
                probas += model.predict_proba(X)
            return probas / len(self.fold_models_)
        else:
            from scipy.stats import mode
            preds = np.array([model.predict(X) for model in self.fold_models_])
            return mode(preds, axis=0)[0].flatten()

    def predict(self, X):
        return self.transform(X) if self.method == 'predict' else self._predict_from_proba(X)

    def predict_proba(self, X):
        if self.method != 'predict_proba':
            raise ValueError("Gunakan method='predict_proba' saat inisialisasi!")
        return self.transform(X)

    def _predict_from_proba(self, X):
        """Helper untuk prediksi kelas dari probabilitas."""
        return np.argmax(self.predict_proba(X), axis=1)

# Recursive Feature Eliminator
class RecursiveFeatureEliminator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_folds=5):
        self.estimator = estimator
        self.n_folds = n_folds
        self.rfe = None

    def fit(self, X, y):
        stratified_kfold = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=42
        )

        self.rfe = RFECV(
            estimator=self.estimator,
            cv=stratified_kfold,
            scoring=self._multiclass_roc_auc_scorer,
            step=1,
            min_features_to_select=1
        )

        self.rfe.fit(X, y)
        return self

    def transform(self, X):
        return self.rfe.transform(X)

    def get_support(self):
        return self.rfe.get_support()

    @staticmethod
    def _multiclass_roc_auc_scorer(estimator, X, y):
        y_pred = estimator.predict_proba(X)
        return roc_auc_score(y, y_pred, multi_class="ovr")

# Classification Evaluator
class ClassificationEvaluator:
    
    def __init__(self, y_true, y_pred, y_proba, label_encoder):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_
        self.num_classes = len(self.classes_)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams['figure.facecolor'] = 'white'

    def plot_confusion_matrix(self, title="Confusion Matrix", figsize=(8, 6)):
        print("Classification Report:")
        print(classification_report(self.y_true, self.y_pred, target_names=self.classes_, digits=6))
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        fig, ax = plt.subplots(figsize=figsize)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes_).plot(
            ax=ax,
            cmap='Greens',
            colorbar=False,
            values_format='d',
            xticks_rotation=45
        )
        
        ax.set_title(title, pad=20, fontsize=14)
        ax.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self, title="ROC Curves (One-vs-Rest)", figsize=(8, 6)):
        y_true_bin = label_binarize(self.y_true, classes=range(self.num_classes))
        fpr, tpr, roc_auc = {}, {}, {}
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        weighted_auc = roc_auc_score(y_true_bin, self.y_proba, multi_class='ovr', average='weighted')
        
        fig, ax = plt.subplots(figsize=figsize)
        for i in range(self.num_classes):
            ax.plot(fpr[i], tpr[i], lw=2, label=f"{self.classes_[i]} (AUC = {roc_auc[i]:.6f})")
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.5)')
        
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, pad=20, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        print(f"\nWeighted Average ROC-AUC: {weighted_auc:.6f}")
        plt.tight_layout()
        plt.show()

    def plot_probability_distribution(self, title='Predicted Probabilities Distribution by True Class', figsize=(10, 6)):
        df_proba = pd.DataFrame(self.y_proba, columns=self.classes_)
        df_proba['TrueLabel'] = self.y_true
        
        plt.figure(figsize=figsize)
        
        for class_idx, class_name in enumerate(self.classes_):
            # Subset for true class examples
            class_data = df_proba[df_proba['TrueLabel'] == class_idx]
            sns.kdeplot(
                data=class_data,
                x=class_name,
                fill=True,
                alpha=0.6,
                label=f'{class_name}',
                linewidth=1
            )
        
        plt.xlabel('Predicted Probability for True Class', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(title, fontsize=14, pad=20)
        plt.legend(title='True Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(False)
        plt.tight_layout()
        plt.show()