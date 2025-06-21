import pandas as pd

from src.basket_model.utils import features
from src.basket_model.utils import loaders
from src.exceptions import UserNotFoundException


class FeatureStore:
    def __init__(self):
        orders = loaders.load_orders()
        regulars = loaders.load_regulars()
        mean_item_price = loaders.get_mean_item_price()
        feature_frame = features.build_feature_frame(orders, regulars, mean_item_price)
        
        # Handle duplicate user IDs by taking the most recent features (latest created_at)
        feature_frame = feature_frame.sort_values('created_at').drop_duplicates('user_id', keep='last')
        
        self.feature_store = (
            feature_frame
            .set_index("user_id")
            .loc[
                :,
                [
                    "prior_basket_value",
                    "prior_item_count",
                    "prior_regulars_count",
                    "regulars_count",
                ],
            ]
        )

    def get_features(self, user_id: str) -> pd.DataFrame:
        try:
            features = self.feature_store.loc[user_id]
        except Exception as exception:
            raise UserNotFoundException(
                "User not found in feature store"
            ) from exception
        return features
