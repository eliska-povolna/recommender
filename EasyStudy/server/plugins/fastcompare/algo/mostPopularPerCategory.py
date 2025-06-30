from abc import ABC
import numpy as np
import pandas as pd
from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


class MostPopularPerCategory(AlgorithmBase, ABC):
    """Implementation of MostPopularPerCategory algorithm."""

    def __init__(self, loader, **kwargs):
        self._loader = loader
        self._ratings_df = loader.ratings_df
        self._categories = loader.get_all_categories()
        all_items = self._ratings_df["item"].unique()
        self._item_categories = {
            item: loader.get_item_index_categories(item) for item in all_items
        }
        self._popularity = self._compute_popularity()

    def _compute_popularity(self):
        """Compute popularity for each item as |users who rated i| / |users|."""
        user_count = self._ratings_df["user"].nunique()
        item_popularity = (
            self._ratings_df.groupby("item")["user"].nunique() / user_count
        )
        return item_popularity

    def fit(self):
        """No fitting required for this algorithm."""
        pass

    def predict(self, selected_items, filter_out_items, k):
        """Generate recommendations of length k."""
        # Filter out items that should not be recommended
        filtered_popularity = self._popularity[
            ~self._popularity.index.isin(filter_out_items)
        ]

        # Sample categories
        print(f"Number of categories: {len(self._categories)}")
        print(self._categories)
        if k <= len(self._categories):
            sampled_categories = np.random.choice(
                list(self._categories), size=k, replace=False
            )
        else:
            sampled_categories = np.random.choice(
                list(self._categories), size=k, replace=True
            )

        recommendations = []
        for category in sampled_categories:
            # Get items in the current category
            items_in_category = [
                item for item, cat in self._item_categories.items() if category in cat
            ]
            # Filter items to ensure they exist in filtered_popularity
            valid_items = [
                item for item in items_in_category if item in filtered_popularity.index
            ]

            if not valid_items:
                continue  # Skip this category if no valid items are found

            # Get the most popular item in this category
            most_popular_item = (
                filtered_popularity.loc[valid_items]
                .sort_values(ascending=False)
                .index[0]
            )
            recommendations.append(most_popular_item)

        return recommendations

    @classmethod
    def name(cls):
        return "MostPopularPerCategory"

    @classmethod
    def parameters(cls):
        return []
