"""
Virus Prediction Model Handler Module
Encapsulates TabularResNet model loading, feature preprocessing, and prediction logic
Uses PyTorch-based neural networks with bundled preprocessing in .pth files
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================================
# VIRUS & SYMPTOM MAPPINGS
# ============================================================================

# Main virus mapping (26 classes)
DEFAULT_VIRUS_MAPPING = {
    0: 'Chikungunya Virus',
    1: 'Dengue Virus',
    2: 'Enterovirus',
    3: 'Hepatitis A Virus',
    4: 'Hepatitis B Virus',
    5: 'Hepatitis C Virus',
    6: 'Hepatitis E Virus',
    7: 'Herpes simplex virus',
    8: 'Influenza A H1N1',
    9: 'Influenza A H3N2',
    10: 'Influenza B Victoria',
    11: 'Japanese Encephalitis',
    12: 'Leptospira',
    13: 'Measles Virus',
    14: 'Mumps Virus',
    15: 'OtherViruses',
    16: 'Parvovirus',
    17: 'Respiratory Adenovirus',
    18: 'Respiratory Syncytial Virus RSV',
    19: 'Respiratory Syncytial Virus-A RSV-A',
    20: 'Respiratory Syncytial Virus-B RSV-B',
    21: 'Rotavirus',
    22: 'Rubella',
    23: 'SARS-Cov-2',
    24: 'Scrub typhus Orientia tsutsugamushi',
    25: 'Varicella zoster virus VZV'
}

# Other Virus sub-classification mapping (13 classes)
DEFAULT_OTHER_VIRUS_MAPPING = {
    0: 'HIV',
    1: 'Haemophilus influenzae',
    2: 'Herpes simplex virus (HSV)',
    3: 'Human papillomavirus (HPV)',
    4: 'Kyasanur Forest Disease',
    5: 'Metapneumovirus',
    6: 'Norovirus',
    7: 'Other Influenza',
    8: 'Rhinovirus',
    9: 'Toxoplasma',
    10: 'Unknown',
    11: 'West Nile virus (WNV)',
    12: 'Zika'
}

VIRUS_MAPPING = dict(DEFAULT_VIRUS_MAPPING)
OTHER_VIRUS_MAPPING = dict(DEFAULT_OTHER_VIRUS_MAPPING)
COMBINED_VIRUS_MAPPING = {}


def _read_virus_mapping_csv(csv_path, expected_count=None):
    df = pd.read_csv(csv_path)
    required_cols = {"Original", "Encoded"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Invalid mapping file: {csv_path}. Expected columns: {required_cols}."
        )

    df = df.dropna(subset=["Original", "Encoded"])
    df["Encoded"] = df["Encoded"].astype(int)
    mapping = dict(zip(df["Encoded"], df["Original"].astype(str)))

    if expected_count is not None and len(mapping) != expected_count:
        st.warning(
            f"Mapping size mismatch for {csv_path}. Expected {expected_count}, got {len(mapping)}."
        )

    return mapping


def refresh_virus_mappings(
    major_csv_path=None,
    other_csv_path=None,
):
    """
    Reload virus name mappings from CSV files and update in place.

    Args:
        major_csv_path: Path to encoding_major_VIRUS_NAME.csv
        other_csv_path: Path to encoding_other_VIRUS_NAME.csv
    """
    base_dir = Path(__file__).resolve().parent
    major_csv_path = major_csv_path or base_dir / "encoding_major_VIRUS_NAME.csv"
    other_csv_path = other_csv_path or base_dir / "encoding_other_VIRUS_NAME.csv"

    major_mapping = dict(DEFAULT_VIRUS_MAPPING)
    other_mapping = dict(DEFAULT_OTHER_VIRUS_MAPPING)

    try:
        if Path(major_csv_path).exists():
            major_mapping = _read_virus_mapping_csv(major_csv_path, expected_count=26)
        else:
            st.warning(f"Major mapping file not found: {major_csv_path}")
    except Exception as exc:
        st.warning(f"Failed to load major mapping from CSV: {exc}")

    try:
        if Path(other_csv_path).exists():
            other_mapping = _read_virus_mapping_csv(other_csv_path, expected_count=13)
        else:
            st.warning(f"Other mapping file not found: {other_csv_path}")
    except Exception as exc:
        st.warning(f"Failed to load other mapping from CSV: {exc}")

    VIRUS_MAPPING.clear()
    VIRUS_MAPPING.update(major_mapping)

    OTHER_VIRUS_MAPPING.clear()
    OTHER_VIRUS_MAPPING.update(other_mapping)

    COMBINED_VIRUS_MAPPING.clear()
    COMBINED_VIRUS_MAPPING.update(
        {f"main_{k}": v for k, v in VIRUS_MAPPING.items() if k != 15}
    )
    COMBINED_VIRUS_MAPPING.update(
        {f"other_{k}": f"Other Viruses → {v}" for k, v in OTHER_VIRUS_MAPPING.items()}
    )


# Combined virus mapping for validation dropdown
refresh_virus_mappings()

# All clinical symptoms (no spaces to match training data)
ALL_SYMPTOMS = [
    'HEADACHE', 'IRRITABLITY', 'ALTEREDSENSORIUM', 'SOMNOLENCE', 'NECKRIGIDITY', 'SEIZURES',
    'DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINALPAIN',
    'MALAISE', 'MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'FEVER',
    'BREATHLESSNESS', 'COUGH', 'RHINORRHEA', 'SORETHROAT',
    'BULLAE', 'PAPULARRASH', 'PUSTULARRASH', 'MUSCULARRASH', 'MACULOPAPULARRASH', 'ESCHAR',
    'DARKURINE', 'HEPATOMEGALY', 'JAUNDICE',
    'REDEYE', 'DISCHARGEEYES', 'CRUSHINGEYES'
]


# ============================================================================
# DEVICE DETECTION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# TABULARRESNET ARCHITECTURE (from notebook)
# ============================================================================

class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(a * F.gelu(b))


class TransformerBlock(nn.Module):
    """Transformer block with gated residual connections"""
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = GEGLU(d_model, d_ff)

        # Simple learnable gating (proven stable)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + torch.sigmoid(self.attn_gate) * self.drop(attn_out)

        h = self.ln2(x)
        ff_out = self.ff(h)
        x = x + torch.sigmoid(self.ff_gate) * self.drop(ff_out)
        return x


class TabularResNet(nn.Module):
    """
    Enhanced TabularResNet for virus classification with:
    - FT-Transformer style continuous feature tokenization
    - Improved projection head for contrastive learning
    - Stable architecture without BatchNorm issues
    """
    def __init__(self, num_binary, num_continuous, cat_dims,
                 num_classes, d_token=256, depth=2, dropout=0.1):
        super().__init__()

        self.num_cat = len(cat_dims)
        self.num_continuous = num_continuous

        # ---------- Categorical Embeddings ----------
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(card, emb) for card, emb in cat_dims
        ])
        self.cat_proj = nn.ModuleList([
            nn.Linear(emb, d_token) for _, emb in cat_dims
        ])

        # ---------- Continuous Features ----------
        # 🔹 IMPROVEMENT: FT-Transformer style tokenization
        # Learnable scaling for better feature representation
        self.cont_proj = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(num_continuous)
        ])
        self.cont_scale = nn.ParameterList([
            nn.Parameter(torch.ones(d_token)) for _ in range(num_continuous)
        ])

        # ---------- Binary Features ----------
        self.bin_linear = nn.Linear(num_binary, d_token)
        self.bin_gate = nn.Parameter(torch.zeros(1))

        # ---------- Token Management ----------
        self.max_tokens = 1 + self.num_cat + num_continuous + (1 if num_binary > 0 else 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, d_token))

        # ---------- Transformer Blocks ----------
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, dropout=dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_token)

        # 🔹 IMPROVEMENT: Enhanced projection head (NO BatchNorm for stability)
        self.proj_head = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
            nn.ReLU(inplace=True),
            nn.Linear(d_token, 128)
        )

        # Classification head
        self.head = nn.Linear(d_token, num_classes)

    def forward(self, xb, xc, xcat, return_embed=False):
        """
        Forward pass through the model.
        
        Args:
            xb: Binary features [batch_size, num_binary]
            xc: Continuous features [batch_size, num_continuous]  
            xcat: Categorical features [batch_size, num_categorical]
            return_embed: If True, return (pooled, z) for contrastive learning
        
        Returns:
            If return_embed: (pooled_features, contrastive_embeddings)
            Else: class_logits
        """
        B = xb.size(0)
        tokens = []

        # ---------- Categorical Tokens ----------
        for i in range(self.num_cat):
            cat_emb = self.cat_embeds[i](xcat[:, i])
            tokens.append(self.cat_proj[i](cat_emb).unsqueeze(1))

        # ---------- Continuous Tokens ----------
        # 🔹 Apply learnable feature-specific scaling
        for i in range(self.num_continuous):
            cont_token = self.cont_proj[i](xc[:, i:i+1])
            cont_token = cont_token * self.cont_scale[i]  # Feature-specific scaling
            tokens.append(cont_token.unsqueeze(1))

        # ---------- Binary Token ----------
        if xb.numel() > 0:
            bin_emb = torch.sigmoid(self.bin_gate) * self.bin_linear(xb)
            tokens.append(bin_emb.unsqueeze(1))

        # ---------- Combine Tokens ----------
        x = torch.cat(tokens, dim=1) if tokens else torch.randn(B, 1, 256).to(DEVICE)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1 + num_tokens, d_token]

        # # ---------- Token Dropout (Training Only) ----------
        # if self.training:
        #     keep = (torch.rand(x.size(1), device=x.device) > 0.1)
        #     keep[0] = True  # Always keep CLS token
        #     x = x[:, keep, :]
        #     pos_embed_used = self.pos_embed[:, :self.max_tokens, :][:, keep, :]
        # else:
        #     pos_embed_used = self.pos_embed[:, :x.size(1), :]

        pos_embed_used = self.pos_embed[:, :x.size(1), :]
        # Add positional embeddings
        x = x + pos_embed_used

        # ---------- Transformer Blocks ----------
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # ---------- Pooling ----------
        # Use proven 70-30 weighted pooling
        pooled = 0.7 * x[:, 0] + 0.3 * x[:, 1:].mean(dim=1)

        # ---------- Return Embeddings or Logits ----------
        if return_embed:
            z = F.normalize(self.proj_head(pooled), dim=1)
            return pooled, z

        return self.head(pooled)


# ============================================================================
# VIRUS PREDICTOR CLASS (using TabularResNet)
# ============================================================================

class VirusPredictor:
    """
    Encapsulates TabularResNet model loading, feature preprocessing, and prediction.
    Loads bundled .pth files with model weights + preprocessing objects.
    """
    
    def __init__(self, model1_path='models/CustomMajor.pth', 
                 model2_path='models/CustomOther.pth'):
        """
        Initialize predictor by loading both pretrained models.
        
        Args:
            model1_path: Path to primary model .pth file (26 major viruses)
            model2_path: Path to secondary model .pth file (13 other virus sub-types)
        """
        self.model1 = None
        self.model2 = None
        self.preprocessing1 = None
        self.preprocessing2 = None
        self.load_models(model1_path, model2_path)
    
    def load_models(self, model1_path, model2_path):
        """
        Load both TabularResNet models with bundled preprocessing.
        
        Args:
            model1_path: Path to primary model
            model2_path: Path to secondary model
        
        Returns:
            bool: True if both models loaded successfully
        """
        try:
            allowlisted = [
                SimpleImputer,
                StandardScaler,
                LabelEncoder,
                np.core.multiarray._reconstruct,
            ]
            torch.serialization.add_safe_globals(allowlisted)

            def _safe_torch_load(path):
                with torch.serialization.safe_globals(allowlisted):
                    try:
                        return torch.load(path, map_location=DEVICE, weights_only=True)
                    except Exception:
                        # st.warning(
                        #     "Safe model load failed; retrying with weights_only=False."
                        #     "Only do this if you trust the checkpoint source."
                        # )
                        return torch.load(path, map_location=DEVICE, weights_only=False)

            # Load Model 1 (Primary - 26 viruses)
            checkpoint1 = _safe_torch_load(model1_path)
            config1 = checkpoint1['model_config']
            self.model1 = TabularResNet(**config1).to(DEVICE)
            self.model1.load_state_dict(checkpoint1['model_state_dict'])
            self.model1.eval()
            self.preprocessing1 = checkpoint1['preprocessing']
            self._normalize_imputer_state(self.preprocessing1)
            
            # Load Model 2 (Secondary - Other Viruses)
            checkpoint2 = _safe_torch_load(model2_path)
            config2 = checkpoint2['model_config']
            self.model2 = TabularResNet(**config2).to(DEVICE)
            self.model2.load_state_dict(checkpoint2['model_state_dict'])
            self.model2.eval()
            self.preprocessing2 = checkpoint2['preprocessing']
            self._normalize_imputer_state(self.preprocessing2)
            
            return True
            
        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}")
            return False
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False

    @staticmethod
    def _normalize_imputer_state(preprocessing):
        """
        Backfill SimpleImputer attributes for cross-version sklearn compatibility.
        Older pickles may miss _fill_dtype, which newer versions access during transform.
        """
        for key in ('imp_cont', 'imp_bin'):
            imputer = preprocessing.get(key)
            if isinstance(imputer, SimpleImputer) and not hasattr(imputer, '_fill_dtype'):
                if hasattr(imputer, '_fit_dtype'):
                    imputer._fill_dtype = imputer._fit_dtype
                elif hasattr(imputer, 'statistics_'):
                    imputer._fill_dtype = np.asarray(imputer.statistics_).dtype
                else:
                    imputer._fill_dtype = np.dtype('float64')
    
    def preprocess_features(self, patient_data, preprocessing):
        """
        Transform patient data dict → binary, continuous, categorical tensors.
        Uses preprocessing objects stored in the model bundle.
        Includes feature engineering to match training data.
        
        Args:
            patient_data: Dictionary with patient demographics and symptoms
            preprocessing: Dict with binary_cols, cat_cols, cont_cols, scalers, encoders
        
        Returns:
            Tuple of (xb, xc, xcat) PyTorch tensors ready for model inference
        """
        try:
            binary_cols = preprocessing['binary_cols']
            cat_cols = preprocessing['cat_cols']
            cont_cols = preprocessing['cont_cols']
            imp_cont = preprocessing['imp_cont']
            scaler = preprocessing['scaler']
            imp_bin = preprocessing['imp_bin']
            le_dict = preprocessing['le_dict']
            
            # Create DataFrame for easier handling
            df = pd.DataFrame([patient_data])
            
            # ========== FEATURE ENGINEERING (to match training) ==========
            
            # 1. AGE FEATURES
            age_median = df['age'].median() if 'age' in df.columns else 30
            df['age'] = df['age'].fillna(age_median).clip(0, 120)
            
            # Age groups
            age_group = pd.cut(df['age'], bins=[0, 5, 18, 45, 65, 150], labels=[0, 1, 2, 3, 4]).cat.codes
            df['age_group'] = age_group.replace(-1, 2)
            
            # 2. SYMPTOM GROUPS & COUNTS
            symptom_cols = [col for col in ALL_SYMPTOMS if col in df.columns]
            respiratory_cols = ['COUGH', 'BREATHLESSNESS', 'RHINORRHEA', 'SORE THROAT']
            gi_cols = ['DIARRHEA', 'DYSENTERY', 'NAUSEA', 'VOMITING', 'ABDOMINAL PAIN']
            neuro_cols = ['HEADACHE', 'ALTERED SENSORIUM', 'SEIZURES', 'SOMNOLENCE', 'NECK RIGIDITY', 'IRRITABILITY']
            skin_cols = ['PAPULAR RASH', 'PUSTULAR RASH', 'MACULOPAPULAR RASH', 'BULLAE']
            systemic_cols = ['MYALGIA', 'ARTHRALGIA', 'CHILLS', 'RIGORS', 'MALAISE']
            
            # Fill missing symptoms with 0
            for col in symptom_cols:
                df[col] = df[col].fillna(0)
            
            # Fill duration
            df['durationofillness'] = df['durationofillness'].fillna(0)
            
            # Create symptom counts
            df['symptom_count'] = df[symptom_cols].sum(axis=1)
            
            # Symptom group counts (handle missing columns)
            resp_present = [c for c in respiratory_cols if c in df.columns]
            df['respiratory_symptoms'] = df[resp_present].sum(axis=1) if resp_present else 0
            
            gi_present = [c for c in gi_cols if c in df.columns]  
            df['gi_symptoms'] = df[gi_present].sum(axis=1) if gi_present else 0
            
            neuro_present = [c for c in neuro_cols if c in df.columns]
            df['neuro_symptoms'] = df[neuro_present].sum(axis=1) if neuro_present else 0
            
            skin_present = [c for c in skin_cols if c in df.columns]
            df['skin_symptoms'] = df[skin_present].sum(axis=1) if skin_present else 0
            
            systemic_present = [c for c in systemic_cols if c in df.columns]
            df['systemic_symptoms'] = df[systemic_present].sum(axis=1) if systemic_present else 0
            
            df['symptom_diversity'] = (df[symptom_cols] > 0).sum(axis=1)
            
            # 3. GEO-TEMPORAL FEATURES
            if 'month' in df.columns:
                # Season mapping
                def get_season(month):
                    if month in [12, 1, 2]: return 0  # Winter
                    elif month in [3, 4, 5]: return 1  # Summer
                    elif month in [6, 7, 8, 9]: return 2  # Monsoon
                    else: return 3  # Post-monsoon
                
                df['season'] = df['month'].apply(get_season)
                
                # Monsoon and winter flags (if not already present)
                if 'is_monsoon' not in df.columns:
                    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
                if 'is_winter' not in df.columns:
                    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
                
                # Cyclical encoding
                if 'month_sin' not in df.columns:
                    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                if 'month_cos' not in df.columns:
                    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                
                # Create missing temporal features (approximate)
                df['week_of_year'] = df['month'] * 4  # Rough approximation
                df['day_of_year'] = df['month'] * 30   # Rough approximation
                df['quarter'] = ((df['month'] - 1) // 3) + 1
            
            # District encoding (create if missing)
            if 'districtencoded' in df.columns and 'district_encoded' not in df.columns:
                df['district_encoded'] = df['districtencoded']
            elif 'district_encoded' not in df.columns:
                df['district_encoded'] = 0  # Default value
                
            # State encoding (ensure correct column name)
            if 'labstate' in df.columns and 'lab_state' not in df.columns:
                df['lab_state'] = df['labstate']
            elif 'lab_state' not in df.columns:
                df['lab_state'] = df.get('labstate', 0)
            
            # Year normalization (if present)
            if 'year' in df.columns:
                # Use 2012-2026 range from training
                df['year_normalized'] = (df['year'] - 2012) / (2026 - 2012)
            else:
                df['year_normalized'] = 0.5  # Default to middle
            
            # 4. INTERACTION FEATURES
            
            # Season-symptom interactions
            if 'season' in df.columns:
                df['monsoon_respiratory'] = df.get('is_monsoon', 0) * df['respiratory_symptoms']
                df['winter_respiratory'] = df.get('is_winter', 0) * df['respiratory_symptoms']
                df['monsoon_fever'] = df.get('is_monsoon', 0) * df.get('FEVER', 0)
                
                # State-season interaction
                df['state_season'] = df['lab_state'] * 10 + df['season']
                
                # District interactions
                df['district_season'] = df['district_encoded'] * 10 + df['season']
                df['district_month'] = df['district_encoded'] * 100 + df.get('month', 1)
            
            # State-symptom interactions
            df['state_respiratory'] = df['lab_state'] * df['respiratory_symptoms']
            df['state_fever'] = df['lab_state'] * df.get('FEVER', 0)
            df['state_gi'] = df['lab_state'] * df['gi_symptoms']
            
            # Other interactions
            df['fever_respiratory'] = df.get('FEVER', 0) * df['respiratory_symptoms']
            df['fever_gi'] = df.get('FEVER', 0) * df['gi_symptoms']
            df['fever_neuro'] = df.get('FEVER', 0) * df['neuro_symptoms'] 
            df['fever_skin'] = df.get('FEVER', 0) * df['skin_symptoms']
            df['fever_duration'] = df.get('FEVER', 0) * df['durationofillness']
            df['fever_headache'] = df.get('FEVER', 0) * df.get('HEADACHE', 0)
            df['fever_cough'] = df.get('FEVER', 0) * df.get('COUGH', 0)
            
            # Severity and complexity features
            df['severity_score'] = df['symptom_count'] * df['durationofillness']
            df['age_symptom'] = df['age'] * df['symptom_count']
            df['age_duration'] = df['age'] * df['durationofillness']
            df['patienttype_age'] = df.get('PATIENTTYPE', 1) * df['age_group']
            df['sex_respiratory'] = df.get('SEX', 1) * df['respiratory_symptoms']
            df['duration_symptom_ratio'] = df['durationofillness'] / (df['symptom_count'] + 1)
            
            # Final cleanup
            df = df.replace([np.inf, -np.inf], 0).fillna(0)
            
            # ========== STANDARD PREPROCESSING ==========
            
            # === CONTINUOUS FEATURES ===
            available_cont_cols = [col for col in cont_cols if col in df.columns]
            if available_cont_cols:
                X_cont = imp_cont.transform(df[available_cont_cols])
                X_cont = scaler.transform(X_cont).astype(np.float32)
            else:
                X_cont = np.zeros((1, len(cont_cols)), dtype=np.float32)
            
            # === BINARY FEATURES ===
            available_bin_cols = [col for col in binary_cols if col in df.columns]
            if available_bin_cols:
                X_bin = imp_bin.transform(df[available_bin_cols]).astype(np.float32)
            else:
                X_bin = np.zeros((1, len(binary_cols)), dtype=np.float32)
            
            # === CATEGORICAL FEATURES ===
            X_cat_list = []
            for col in cat_cols:
                if col in df.columns:
                    le = le_dict[col]
                    val = str(df[col].values[0])
                    mapping = dict(zip(le.classes_, range(len(le.classes_))))
                    encoded_val = mapping.get(val, 0)
                    X_cat_list.append(encoded_val)
                else:
                    X_cat_list.append(0)  # Default value for missing categorical
            
            X_cat = np.array([X_cat_list], dtype=np.int64) if cat_cols else np.zeros((1, 0), dtype=np.int64)
            
            # Convert to PyTorch tensors
            xb = torch.tensor(X_bin, dtype=torch.float32).to(DEVICE)
            xc = torch.tensor(X_cont, dtype=torch.float32).to(DEVICE)
            xcat = torch.tensor(X_cat, dtype=torch.long).to(DEVICE)
            
            return xb, xc, xcat
            
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            import traceback
            st.error(traceback.format_exc())
            raise
    
    def predict(self, patient_data):
        """
        Complete prediction workflow: preprocess features and run both models.
        
        Args:
            patient_data: Dictionary with patient information
        
        Returns:
            dict: Prediction results with probabilities from both models
        """
        if self.model1 is None or self.model2 is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        try:
            # Preprocess for Model 1
            xb1, xc1, xcat1 = self.preprocess_features(patient_data, self.preprocessing1)
            
            # Model 1 prediction (26 major viruses)
            with torch.no_grad():
                logits1 = self.model1(xb1, xc1, xcat1)  # Don't use return_embed for inference
                y_pred_proba = torch.softmax(logits1, dim=1)[0].cpu().numpy()
            
            y_pred = np.argmax(y_pred_proba)
            top_5_indices = np.argsort(y_pred_proba)[-5:][::-1]
            
            # Check if "Other_Viruses" (class 15) is in top 5
            second_model_results = None
            if 15 in top_5_indices:
                xb2, xc2, xcat2 = self.preprocess_features(patient_data, self.preprocessing2)
                
                with torch.no_grad():
                    logits2 = self.model2(xb2, xc2, xcat2)  # Don't use return_embed for inference
                    y_pred_proba_m2 = torch.softmax(logits2, dim=1)[0].cpu().numpy()
                
                y_pred_m2 = np.argmax(y_pred_proba_m2)
                top_5_indices_m2 = np.argsort(y_pred_proba_m2)[-5:][::-1]
                
                second_model_results = {
                    'prediction': y_pred_m2,
                    'probabilities': y_pred_proba_m2,
                    'top_5': top_5_indices_m2
                }
            
            return {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'top_5_indices': top_5_indices,
                'second_model_results': second_model_results
            }
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            raise


# ============================================================================
# STREAMLIT CACHE WRAPPER (for use in Streamlit apps)
# ============================================================================

@st.cache_resource
def get_virus_predictor():
    """
    Get or create a cached VirusPredictor instance (for Streamlit caching).
    
    Returns:
        VirusPredictor: Initialized predictor with loaded models
    """
    return VirusPredictor()
