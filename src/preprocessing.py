from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DataConfig, SplitRatios
from src.constants import PROJECT_ROOT
from src.dataset import TabularSplit


@dataclass(frozen=True)
class TabularSplitsCollection:
    train: TabularSplit
    val: TabularSplit
    test: TabularSplit

    def to_csv(self, export_dir: Union[str, Path]) -> None:
        self.train.to_csv(export_dir)
        self.val.to_csv(export_dir)
        self.test.to_csv(export_dir)


def _read_data(csv_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(PROJECT_ROOT / csv_path)
    features = data.drop([target_col], axis=1)
    target = data[target_col]
    return features, target


def _split_data(
    features: pd.DataFrame,
    target: pd.Series,
    split_ratios: SplitRatios,
    seed: int,
) -> TabularSplitsCollection:
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        features,
        target,
        stratify=target,
        test_size=(1 - split_ratios.train),
        random_state=seed,
    )

    relative_test_ratio = split_ratios.test / (split_ratios.val + split_ratios.test)
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test,
        y_val_test,
        stratify=y_val_test,
        test_size=relative_test_ratio,
        random_state=seed,
    )

    train_split = TabularSplit(x_train, y_train, 'train')
    val_split = TabularSplit(x_val, y_val, 'val')
    test_split = TabularSplit(x_test, y_test, 'test')

    return TabularSplitsCollection(train_split, val_split, test_split)


def _fit_col_transformer(
    features: pd.DataFrame,
    categorical_cols: Optional[Tuple[str, ...]] = None,
    apply_standardization: bool = False,
) -> ColumnTransformer:
    transformers = []

    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))

        numeric_cols = tuple(set(features.columns) - set(categorical_cols))
    else:
        numeric_cols = tuple(features.columns)

    if numeric_cols and apply_standardization:
        transformers.append(('num', StandardScaler(), numeric_cols))

    transformer = ColumnTransformer(transformers=transformers, remainder='passthrough', verbose_feature_names_out=False)
    transformer.fit(features)

    return transformer


def _filter_positive_cols(
    features: pd.DataFrame,
    target: pd.Series,
    positive_cols: Optional[Tuple[str, ...]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if not positive_cols:
        return features, target
    masks = []
    for col in positive_cols:
        mask = features[col] > 0
        masks.append(mask)
    positives_bool = np.prod(np.vstack(masks), axis=0).astype(bool)
    return features[positives_bool], target[positives_bool]


def _np_to_df(features_np: np.ndarray, col_names: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(features_np, columns=col_names)


def _transform_cols(transformer: ColumnTransformer, splits: TabularSplitsCollection) -> TabularSplitsCollection:
    train_features_np = transformer.transform(splits.train.features)
    val_features_np = transformer.transform(splits.val.features)
    test_features_np = transformer.transform(splits.test.features)

    col_names = transformer.get_feature_names_out()
    np_to_df = partial(_np_to_df, col_names=col_names)

    train_features = np_to_df(train_features_np)
    val_features = np_to_df(val_features_np)
    test_features = np_to_df(test_features_np)

    return TabularSplitsCollection(
        train=TabularSplit(train_features, splits.train.target, 'train'),
        val=TabularSplit(val_features, splits.val.target, 'val'),
        test=TabularSplit(test_features, splits.test.target, 'test'),
    )


def preprocess_data(cfg: DataConfig) -> None:
    print(f'Initiating dataset preprocessing using the following config: {cfg}')
    lightning.seed_everything(cfg.seed)

    features, target = _read_data(cfg.raw_csv_path, cfg.target_column)
    features, target = _filter_positive_cols(features, target, cfg.positive_columns)
    splits = _split_data(features, target, cfg.split_ratios, cfg.seed)

    transformer = _fit_col_transformer(splits.train.features, cfg.categorical_columns, cfg.apply_standardization)
    splits = _transform_cols(transformer, splits)

    splits.to_csv(cfg.processed_path)

    print(f'Dataset is preprocessed and saved to {cfg.processed_path}')
