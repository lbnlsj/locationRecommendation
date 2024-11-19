# data_structures.py
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime


@dataclass
class POIFeature:
    venue_id: str
    category_id: str
    category: str
    coordinates: Tuple[float, float]
    temporal_pattern: np.ndarray
    popularity: float


class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.visit_history: List[Tuple[str, datetime]] = []
        self.category_preferences: Dict[str, float] = {}
        self.spatial_radius: float = None
        self.active_hours: np.ndarray = np.zeros(24)
        self.active_days: np.ndarray = np.zeros(7)


# data_processor.py
import pandas as pd
from sklearn.preprocessing import normalize
import torch
from torch.utils.data import Dataset


class CheckinDataProcessor:
    def __init__(self):
        self.venue_data = {}
        self.user_profiles = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'])
        return df

    def extract_temporal_features(self, df: pd.DataFrame, venue_id: str) -> np.ndarray:
        venue_visits = df[df['venueId'] == venue_id]
        temporal_matrix = np.zeros((24, 7))
        for _, row in venue_visits.iterrows():
            hour = row['utcTimestamp'].hour
            day = row['utcTimestamp'].dayofweek
            temporal_matrix[hour, day] += 1
        return normalize(temporal_matrix, axis=1, norm='l2')

    def create_user_profiles(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            if row['userId'] not in self.user_profiles:
                self.user_profiles[row['userId']] = UserProfile(row['userId'])

            profile = self.user_profiles[row['userId']]
            profile.visit_history.append((row['venueId'], row['utcTimestamp']))

            cat = row['venueCategoryId']
            profile.category_preferences[cat] = profile.category_preferences.get(cat, 0) + 1


# models.py
import torch.nn as nn
import torch.nn.functional as F


class POIEncoder(nn.Module):
    def __init__(self, n_categories: int, hidden_dim: int):
        super().__init__()
        self.category_embedding = nn.Embedding(n_categories, hidden_dim)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.temporal_encoder = nn.Sequential(
            nn.Linear(24 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, poi_feature: POIFeature) -> torch.Tensor:
        cat_emb = self.category_embedding(torch.tensor(int(poi_feature.category_id)))
        spatial_emb = self.spatial_encoder(torch.tensor(poi_feature.coordinates))
        temporal_emb = self.temporal_encoder(torch.tensor(poi_feature.temporal_pattern.flatten()))

        combined = torch.cat([cat_emb, spatial_emb, temporal_emb], dim=-1)
        return self.fusion_layer(combined)


class RewardModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.poi_encoder = POIEncoder(config['n_categories'], config['hidden_dim'])
        self.temporal_scorer = nn.Linear(config['hidden_dim'], 1)
        self.spatial_scorer = nn.Linear(config['hidden_dim'], 1)
        self.preference_scorer = nn.Linear(config['hidden_dim'], 1)
        self.weight_layer = nn.Linear(config['hidden_dim'], 3)

    def forward(self, poi_feature: POIFeature, user_profile: UserProfile,
                context: Dict) -> torch.Tensor:
        poi_embedding = self.poi_encoder(poi_feature)

        temporal_score = self.temporal_scorer(poi_embedding)
        spatial_score = self.spatial_scorer(poi_embedding)
        preference_score = self.preference_scorer(poi_embedding)

        # Dynamic weight computation
        weights = F.softmax(self.weight_layer(poi_embedding), dim=-1)

        scores = torch.stack([temporal_score, spatial_score, preference_score])
        final_score = (weights * scores).sum()

        return final_score


# trainer.py
from torch.optim import AdamW
from torch.utils.data import DataLoader


class RewardModelTrainer:
    def __init__(self, model: RewardModel, config: Dict):
        self.model = model
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=config['lr'])

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            pos_score = self.model(batch['pos_poi'], batch['user'], batch['context'])
            neg_score = self.model(batch['neg_poi'], batch['user'], batch['context'])

            loss = torch.max(torch.zeros_like(pos_score),
                             neg_score - pos_score + self.config['margin'])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)


# main.py
def main():
    config = {
        'n_categories': 1000,
        'hidden_dim': 256,
        'lr': 1e-4,
        'margin': 0.5,
        'epochs': 10,
    }

    # Initialize data processor
    processor = CheckinDataProcessor()
    df = processor.load_data('checkins.csv')
    processor.create_user_profiles(df)

    # Initialize model
    model = RewardModel(config)
    trainer = RewardModelTrainer(model, config)

    # Training loop
    for epoch in range(config['epochs']):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


if __name__ == "__main__":
    main()