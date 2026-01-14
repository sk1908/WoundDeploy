"""
Risk Prediction Model
Predicts non-healing likelihood from wound features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import risk_config, DEVICE
from .features import WoundFeatures, FeatureExtractor


@dataclass
class RiskResult:
    """Risk prediction result"""
    risk_score: float  # 0-1 probability
    risk_level: str    # categorical: low, moderate, high, critical
    risk_factors: Dict[str, float]  # contribution of each feature
    recommendations: List[str]  # action items


class RiskModel(nn.Module):
    """
    MLP model for non-healing risk prediction
    """
    
    def __init__(self,
                 input_features: int = None,
                 hidden_dims: List[int] = None,
                 dropout: float = None):
        super().__init__()
        
        self.input_features = input_features or risk_config.input_features
        hidden_dims = hidden_dims or risk_config.hidden_dims
        dropout = dropout or risk_config.dropout
        
        # Build MLP
        layers = []
        prev_dim = self.input_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        # Feature importance weights (learned)
        self.feature_weights = nn.Parameter(torch.ones(self.input_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Feature tensor (B, input_features)
            
        Returns:
            Risk score (B, 1) in [0, 1]
        """
        # Apply feature weighting
        weighted_x = x * F.softmax(self.feature_weights, dim=0)
        return self.mlp(weighted_x)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance weights"""
        return F.softmax(self.feature_weights, dim=0).detach().cpu().numpy()


class RiskPredictor:
    """
    High-level risk prediction interface
    """
    
    def __init__(self,
                 weights_path: Optional[Path] = None,
                 device: Optional[torch.device] = None):
        self.weights_path = weights_path or risk_config.weights_path
        self.device = device or DEVICE
        
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self._is_loaded = False
        
        # Feature names for interpretability
        self.feature_names = [
            "granulation_ratio", "slough_ratio", "necrotic_ratio", "epithelium_ratio",
            "wound_area", "wound_perimeter", "circularity",
            "mean_depth", "max_depth", "depth_variance",
            "total_volume", "necrotic_volume",
            "healthy_ratio", "necrotic_burden"
        ]
        
        # Risk thresholds
        self.risk_thresholds = {
            "low": 0.25,
            "moderate": 0.50,
            "high": 0.75
        }
        
        # Heuristic weights (used when model not available)
        self.heuristic_weights = {
            "necrotic_ratio": 0.25,
            "slough_ratio": 0.15,
            "necrotic_burden": 0.20,
            "wound_area": 0.10,
            "max_depth": 0.10,
            "healthy_ratio": -0.20  # Negative = protective
        }
    
    def load(self) -> bool:
        """Load trained model"""
        try:
            self.model = RiskModel()
            
            if self.weights_path.exists():
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded risk model from {self.weights_path}")
            else:
                print("No trained risk model, using heuristic scoring")
            
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            
            return True
        except Exception as e:
            print(f"Error loading risk model: {e}")
            return False
    
    def predict(self,
                features: WoundFeatures = None,
                segmentation_mask: np.ndarray = None,
                depth_map: np.ndarray = None,
                use_model: bool = True) -> RiskResult:
        """
        Predict non-healing risk
        
        Args:
            features: Pre-extracted features (optional)
            segmentation_mask: Segmentation mask (optional if features provided)
            depth_map: Depth map (optional)
            use_model: Use ML model vs heuristics
            
        Returns:
            RiskResult with score, level, factors, and recommendations
        """
        # Extract features if not provided
        if features is None:
            if segmentation_mask is None:
                return RiskResult(
                    risk_score=0.5,
                    risk_level="unknown",
                    risk_factors={},
                    recommendations=["Insufficient data for risk assessment"]
                )
            features = self.feature_extractor.extract(segmentation_mask, depth_map)
        
        # Compute risk score
        if use_model and self._is_loaded and self.weights_path.exists():
            risk_score = self._predict_with_model(features)
            feature_importance = self._get_model_importance(features)
        else:
            risk_score = self._compute_heuristic_score(features)
            feature_importance = self._get_heuristic_importance(features)
        
        # Determine risk level
        risk_level = self._score_to_level(risk_score)
        
        # Get top risk factors
        risk_factors = self._identify_risk_factors(features, feature_importance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(features, risk_level)
        
        return RiskResult(
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations
        )
    
    def _predict_with_model(self, features: WoundFeatures) -> float:
        """Predict using the trained model"""
        feature_array = features.to_array()
        tensor = torch.from_numpy(feature_array).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.model(tensor).item()
        
        return score
    
    def _compute_heuristic_score(self, features: WoundFeatures) -> float:
        """Compute risk using heuristic rules"""
        score = 0.3  # Base risk
        
        feature_dict = features.to_dict()
        
        for feature_name, weight in self.heuristic_weights.items():
            if feature_name in feature_dict:
                value = feature_dict[feature_name]
                
                # Normalize values if needed
                if feature_name == "wound_area":
                    value = min(value / 10000, 1.0)  # Normalize large areas
                elif feature_name == "max_depth":
                    value = min(value, 1.0)
                
                score += weight * value
        
        # Additional rules
        if features.necrotic_ratio > 0.3:
            score += 0.15
        if features.healthy_ratio < 0.2:
            score += 0.10
        if features.wound_area > 5000:
            score += 0.05
        
        return np.clip(score, 0.0, 1.0)
    
    def _get_model_importance(self, features: WoundFeatures) -> Dict[str, float]:
        """Get feature importance from model"""
        if self.model is None:
            return {}
        
        importance = self.model.get_feature_importance()
        return dict(zip(self.feature_names, importance))
    
    def _get_heuristic_importance(self, features: WoundFeatures) -> Dict[str, float]:
        """Get feature importance from heuristics"""
        # Use absolute weights as importance
        total = sum(abs(w) for w in self.heuristic_weights.values())
        return {k: abs(v) / total for k, v in self.heuristic_weights.items()}
    
    def _score_to_level(self, score: float) -> str:
        """Convert risk score to categorical level"""
        if score < self.risk_thresholds["low"]:
            return "low"
        elif score < self.risk_thresholds["moderate"]:
            return "moderate"
        elif score < self.risk_thresholds["high"]:
            return "high"
        else:
            return "critical"
    
    def _identify_risk_factors(self,
                                features: WoundFeatures,
                                importance: Dict[str, float]) -> Dict[str, float]:
        """Identify top contributing risk factors"""
        feature_dict = features.to_dict()
        
        # Score each feature's contribution
        contributions = {}
        for name, imp in importance.items():
            if name in feature_dict:
                value = feature_dict[name]
                # High necrotic/slough is bad, high healthy is good
                if name in ["necrotic_ratio", "slough_ratio", "necrotic_burden"]:
                    contributions[name] = imp * value
                elif name == "healthy_ratio":
                    contributions[name] = imp * (1 - value)  # Low healthy = high risk
                else:
                    contributions[name] = imp * min(value, 1.0)
        
        # Return top 5
        sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_factors[:5])
    
    def _generate_recommendations(self,
                                    features: WoundFeatures,
                                    risk_level: str) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        # Based on risk level
        if risk_level == "critical":
            recommendations.append("⚠️ Urgent medical attention recommended")
            recommendations.append("Consider surgical debridement evaluation")
        elif risk_level == "high":
            recommendations.append("Schedule follow-up within 2-3 days")
            recommendations.append("Consider advanced wound care therapies")
        
        # Based on specific features
        if features.necrotic_ratio > 0.2:
            recommendations.append("Debridement may be indicated for necrotic tissue")
        
        if features.slough_ratio > 0.3:
            recommendations.append("Consider autolytic or enzymatic debridement for slough")
        
        if features.healthy_ratio < 0.3:
            recommendations.append("Promote granulation tissue formation")
        
        if features.wound_area > 5000:
            recommendations.append("Large wound area - consider compression or negative pressure therapy")
        
        if features.max_depth > 0.5:
            recommendations.append("Deep wound - monitor for tunneling or undermining")
        
        # General recommendation
        if risk_level in ["low", "moderate"]:
            recommendations.append("Continue current treatment protocol with regular monitoring")
        
        return recommendations[:5]  # Limit to top 5


def train_risk_model(features_path: Path, labels_path: Path, output_path: Path):
    """
    Train risk model on labeled data
    
    Note: In practice, you'd need clinical outcome data (healed vs non-healed)
    """
    import json
    
    # Load data
    with open(features_path) as f:
        features_data = json.load(f)
    
    with open(labels_path) as f:
        labels_data = json.load(f)
    
    # Prepare training data
    X = np.array([f["features"] for f in features_data])
    y = np.array([l["non_healing"] for l in labels_data])  # 0 or 1
    
    # Create model
    model = RiskModel(input_features=X.shape[1])
    model.to(DEVICE)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X).float().to(DEVICE)
    y_tensor = torch.from_numpy(y).float().unsqueeze(1).to(DEVICE)
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    # Test risk prediction
    import cv2
    
    # Create dummy data
    mask = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    depth = np.random.rand(256, 256).astype(np.float32)
    
    predictor = RiskPredictor()
    result = predictor.predict(segmentation_mask=mask, depth_map=depth, use_model=False)
    
    print(f"\nRisk Assessment:")
    print(f"  Score: {result.risk_score:.2f}")
    print(f"  Level: {result.risk_level}")
    print(f"\nTop Risk Factors:")
    for factor, contribution in result.risk_factors.items():
        print(f"  {factor}: {contribution:.3f}")
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  • {rec}")
