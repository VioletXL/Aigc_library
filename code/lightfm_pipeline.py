#!/usr/bin/env python3
"""LightFM推荐模型训练脚本"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd

try:
    from lightfm import LightFM
    from lightfm.data import Dataset
    from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
except ImportError as exc:
    raise ImportError(
        "lightfm package is required. Install it with 'pip install lightfm' before running this script."
    ) from exc

DEFAULT_RANDOM_STATE = 42


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame


class LightFMPipeline:
    """LightFM模型训练和评估"""

    def __init__(
        self,
        interactions_path: str,
        items_path: str,
        users_path: str,
        test_ratio: float = 0.2,
        min_test_items: int = 1,
        loss: str = "warp",
        no_components: int = 64,
        learning_rate: float = 0.05,
        epochs: int = 30,
        num_threads: int = 4,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.interactions_path = interactions_path
        self.items_path = items_path
        self.users_path = users_path
        self.test_ratio = test_ratio
        self.min_test_items = min_test_items
        self.loss = loss
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_threads = num_threads
        self.random_state = random_state

        self.dataset: Dataset | None = None
        self.model: LightFM | None = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        interactions = pd.read_csv(self.interactions_path)
        interactions = interactions.rename(columns={"借阅时间": "borrow_time"})
        interactions["borrow_time"] = pd.to_datetime(interactions["borrow_time"])
        interactions["user_id"] = interactions["user_id"].astype(str)
        interactions["book_id"] = interactions["book_id"].astype(str)

        item_meta = pd.read_csv(self.items_path).rename(columns={"book_id": "book_id"})
        item_meta["book_id"] = item_meta["book_id"].astype(str)

        user_meta = pd.read_csv(self.users_path).rename(columns={"借阅人": "user_id"})
        user_meta["user_id"] = user_meta["user_id"].astype(str)

        return interactions, item_meta, user_meta

    def temporal_split(self, interactions: pd.DataFrame) -> SplitResult:
        # interactions to the test set.
        interactions = interactions.sort_values(["user_id", "borrow_time"]).copy()

        train_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []

        for user_id, group in interactions.groupby("user_id"):
            total = len(group)
            if total <= self.min_test_items:
                train_parts.append(group)
                continue

            test_size = max(self.min_test_items, int(np.ceil(total * self.test_ratio)))
            train_size = total - test_size
            if train_size <= 0:
                train_size = total - 1
                test_size = 1

            train_parts.append(group.iloc[:train_size])
            test_parts.append(group.iloc[train_size:])

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(
            columns=interactions.columns
        )

        return SplitResult(train=train_df, test=test_df)

    @staticmethod
    def _clean_value(value: object, prefix: str) -> str:
        if pd.isna(value) or value == "":
            return f"{prefix}:unknown"
        return f"{prefix}:{value}"

    def _collect_user_features(self, user_meta: pd.DataFrame) -> Tuple[set, List[Tuple[str, List[str]]]]:
        feature_terms: set = set()
        feature_rows: List[Tuple[str, List[str]]] = []
        for row in user_meta.itertuples(index=False):
            features = [
                self._clean_value(row.性别, "gender"),
                self._clean_value(row.DEPT, "dept"),
                self._clean_value(row.年级, "grade"),
                self._clean_value(row.类型, "type"),
            ]
            feature_terms.update(features)
            feature_rows.append((str(row.user_id), features))
        return feature_terms, feature_rows

    def _collect_item_features(self, item_meta: pd.DataFrame) -> Tuple[set, List[Tuple[str, List[str]]]]:
        feature_terms: set = set()
        feature_rows: List[Tuple[str, List[str]]] = []
        for row in item_meta.itertuples(index=False):
            features = [
                self._clean_value(row.作者, "author"),
                self._clean_value(row.出版社, "press"),
                self._clean_value(row.一级分类, "cat1"),
                self._clean_value(row.二级分类, "cat2"),
            ]
            feature_terms.update(features)
            feature_rows.append((str(row.book_id), features))
        return feature_terms, feature_rows

    def _ensure_metadata(
        self,
        ids: Iterable,
        meta_df: pd.DataFrame,
        id_column: str,
        defaults: dict,
    ) -> pd.DataFrame:
        existing = set(meta_df[id_column].astype(str))
        missing = [str(i) for i in ids if str(i) not in existing]
        if not missing:
            return meta_df

        fillers = pd.DataFrame([
            {id_column: m, **defaults} for m in missing
        ])
        return pd.concat([meta_df, fillers], ignore_index=True)

    def _build_dataset(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        user_meta: pd.DataFrame,
        item_meta: pd.DataFrame,
    ) -> Tuple:
        dataset = Dataset()

        all_users_series = pd.concat(
            [
                train_df["user_id"],
                test_df["user_id"] if not test_df.empty else pd.Series(dtype=train_df["user_id"].dtype),
            ]
        )
        all_items_series = pd.concat(
            [
                train_df["book_id"],
                test_df["book_id"] if not test_df.empty else pd.Series(dtype=train_df["book_id"].dtype),
            ]
        )
        all_users = all_users_series.astype(str).unique()
        all_items = all_items_series.astype(str).unique()

        user_meta = self._ensure_metadata(
            all_users,
            user_meta,
            id_column="user_id",
            defaults={"性别": np.nan, "DEPT": np.nan, "年级": np.nan, "类型": np.nan},
        )
        item_meta = self._ensure_metadata(
            all_items,
            item_meta.rename(columns={"book_id": "book_id"}),
            id_column="book_id",
            defaults={"题名": np.nan, "作者": np.nan, "出版社": np.nan, "一级分类": np.nan, "二级分类": np.nan},
        )

        user_meta = user_meta[user_meta["user_id"].isin(all_users)].copy()
        item_meta = item_meta[item_meta["book_id"].isin(all_items)].copy()

        user_terms, user_features = self._collect_user_features(user_meta)
        item_terms, item_features = self._collect_item_features(item_meta)

        dataset.fit(
            (str(u) for u in all_users),
            (str(i) for i in all_items),
            user_features=user_terms,
            item_features=item_terms,
        )

        (train_interactions, _) = dataset.build_interactions(
            (str(u), str(i), 1.0) for u, i in train_df[["user_id", "book_id"]].itertuples(index=False)
        )
        (test_interactions, _) = dataset.build_interactions(
            (str(u), str(i), 1.0) for u, i in test_df[["user_id", "book_id"]].itertuples(index=False)
        )

        user_features_matrix = dataset.build_user_features(user_features)
        item_features_matrix = dataset.build_item_features(item_features)

        self.dataset = dataset
        return train_interactions, test_interactions, user_features_matrix, item_features_matrix

    def train(self, train_interactions, user_features, item_features) -> None:
        self.model = LightFM(
            loss=self.loss,
            no_components=self.no_components,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.model.fit(
            train_interactions,
            user_features=user_features,
            item_features=item_features,
            epochs=self.epochs,
            num_threads=self.num_threads,
        )

    def evaluate(self, interactions, user_features, item_features) -> Tuple[float, float]:
        if self.model is None:
            raise RuntimeError("Model has not been trained")
        prec = precision_at_k(
            self.model,
            interactions,
            user_features=user_features,
            item_features=item_features,
            k=10,
            num_threads=self.num_threads,
        ).mean()
        auc = auc_score(
            self.model,
            interactions,
            user_features=user_features,
            item_features=item_features,
            num_threads=self.num_threads,
        ).mean()
        return prec, auc

    def recall(self, interactions, user_features, item_features, k: int = 10) -> float:
        if self.model is None:
            raise RuntimeError("Model has not been trained")
        return recall_at_k(
            self.model,
            interactions,
            user_features=user_features,
            item_features=item_features,
            k=k,
            num_threads=self.num_threads,
        ).mean()

    # ------------------------------------------------------------------
    # End-to-end run
    # ------------------------------------------------------------------
    def run(self) -> None:
        interactions, item_meta, user_meta = self.load_data()
        split = self.temporal_split(interactions)

        print(f"Total interactions: {len(interactions)}")
        print(f"Train interactions: {len(split.train)} | Test interactions: {len(split.test)}")

        train_matrix, test_matrix, user_features, item_features = self._build_dataset(
            split.train, split.test, user_meta, item_meta
        )

        self.train(train_matrix, user_features, item_features)
        train_prec, train_auc = self.evaluate(train_matrix, user_features, item_features)
        if test_matrix.nnz > 0:
            test_prec, test_auc = self.evaluate(test_matrix, user_features, item_features)
            test_recall = self.recall(test_matrix, user_features, item_features)
        else:
            test_prec = test_auc = test_recall = float("nan")

        print("\n=== Evaluation ===")
        print(f"Train precision@10: {train_prec:.4f} | Train AUC: {train_auc:.4f}")
        if np.isnan(test_prec):
            print("Test precision@10 : N/A (test split empty)")
            print("Test AUC          : N/A")
            print("Test recall@10    : N/A")
        else:
            print(f"Test precision@10 : {test_prec:.4f} | Test AUC : {test_auc:.4f}")
            print(f"Test recall@10    : {test_recall:.4f}")

        export_path = os.path.join("output", "lightfm_model.npz")
        self.save_model(export_path)
        print(f"Model saved to {export_path}")

    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been trained")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            user_embeddings=self.model.user_embeddings,
            item_embeddings=self.model.item_embeddings,
            user_biases=self.model.user_biases,
            item_biases=self.model.item_biases,
        )

        if self.dataset is None:
            raise RuntimeError("Dataset metadata is unavailable")

        user_mapping = getattr(self.dataset, "_user_id_mapping", {})
        item_mapping = getattr(self.dataset, "_item_id_mapping", {})

        user_ids = [uid for uid, _ in sorted(user_mapping.items(), key=lambda kv: kv[1])]
        item_ids = [iid for iid, _ in sorted(item_mapping.items(), key=lambda kv: kv[1])]

        mapping = {"user_ids": user_ids, "item_ids": item_ids}
        mapping_path = os.path.splitext(path)[0] + "_mappings.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightFM with metadata features")
    parser.add_argument("--interactions", default="data/inter_final_选手可见.csv")
    parser.add_argument("--items", default="data/item.csv")
    parser.add_argument("--users", default="data/user.csv")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--loss", choices=["warp", "bpr", "warp-kos", "logistic"], default="warp")
    parser.add_argument("--components", type=int, default=64, help="Latent dimensionality")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = LightFMPipeline(
        interactions_path=args.interactions,
        items_path=args.items,
        users_path=args.users,
        test_ratio=args.test_ratio,
        loss=args.loss,
        no_components=args.components,
        learning_rate=args.lr,
        epochs=args.epochs,
        num_threads=args.threads,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
