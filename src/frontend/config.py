from src.global_settings import settings

VERSION = "1.1"

# ============================================================================
# Path Configuration
# ============================================================================
PROJECT_ROOT = settings.project_root
FRONTEND_ROOT = PROJECT_ROOT / "src" / "frontend"

TEXT_DATASET_PATH = settings.text_dataset_path
RESIDUAL_STREAM_DIR = settings.rs_data_dir
LM_WEIGHTS_PATH = settings.lm_weights_path
SAE_WEIGHTS_DIR = PROJECT_ROOT / "outputs" / "weights"
FEATURE_RECORD_DIR = settings.feature_record_dir
FEATURE_META_DIR = settings.feature_meta_dir

# ============================================================================
# Model Configuration
# ============================================================================
ALLOWED_MODEL_TAGS = [
    "f960_k8",
    "f3072_k8",
    "f3072_k256",
    "f6144_k32",
    "f6144_k256",
    "f6144_k1024",
]
ALLOWED_LAYERS = settings.layers
ALLOWED_HPARAMSET_TAGS = [
    "alpha",
]
ALLOWED_DOMAINS = [
    "default",
]
DEFAULT_MODEL_TAG = "f3072_k8"
DEFAULT_LAYER = 21
DEFAULT_HPARAMSET_TAG = "alpha"
DEFAULT_DOMAIN = "default"
DEFAULT_FEATURE_INDEX = 6
MIN_FEATURE_INDEX = 0
MAX_FEATURE_INDEX = {
    "f960_k8": 960,
    "f3072_k8": 3072,
    "f3072_k256": 3072,
    "f6144_k32": 6144,
    "f6144_k256": 6144,
    "f6144_k1024": 6144,
}  # e.g., for f960_k8, valid feature indices are 0 to 959

# ============================================================================
# UI Configuration
# ============================================================================
APP_TITLE = "SAE Explorer"
APP_ICON = "🔮"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "collapsed"

TOP_MODULES = [
    {
        "id": "about",
        "name": "About",
        "description": "Learn how SAE and SAE Explorer works",
        "icon": "ℹ️",
        "color": "#E5F3FF",
        "requires_settings": False,
        "disabled": False,
    },
    {
        "id": "features",
        "name": "Features",
        "description": "Explore all potential features",
        "icon": "🔬",
        "color": "#E5FFE5",
        "requires_settings": True,
        "disabled": False,
    },
    {
        "id": "inference",
        "name": "Inference",
        "description": "Test feature decoding with custom text",
        "icon": "✍🏻",
        "color": "#FFF5E5",
        "requires_settings": True,
        "disabled": False,
    },
    {
        "id": "steering",
        "name": "Steering",
        "description": "Change the classifier’s prediction by manipulating features",
        "icon": "🕹️",
        "color": "#FFE5E5",
        "requires_settings": True,
        "disabled": False,
    },
    # {
    #     "id": "experiments",
    #     "name": "Experiments",
    #     "description": "Explore more on-going experiments",
    #     "icon": "🚀",
    #     "color": "#F5E5FF",
    #     "requires_settings": False,
    #     "disabled": True,
    # },
]
MODULE_MAP = {module["id"]: module for module in TOP_MODULES}
