import os
import torch
from app.utils.logging import logger
from app.models.aptos_model import load_aptos_keras_model
from app.models.vessel_model import DriveUNet
from app.models.idrid_model import IdridUNet

class ModelRegistry:
    """
    Centralized singleton registry for managing pre-trained AI models.
    Loads models once into memory during application startup.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, weights_dir: str = None):
        if self._initialized:
            return

        if weights_dir is None:
            weights_dir = os.getenv("MODEL_WEIGHTS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "weights"))
        weights_dir = os.path.abspath(weights_dir)
        logger.info(f"Initializing ModelRegistry from weights directory: {weights_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch computing device: {self.device}")

        # 1. Model 1: APTOS DR Classifier
        self.aptos_model = None
        aptos_path = os.path.join(weights_dir, "best_aptos_model.keras")
        try:
            self.aptos_model = load_aptos_keras_model(aptos_path)
            logger.info("Model 1 (APTOS) successfully loaded into memory.")
        except Exception as e:
            logger.error(f"Failed to load Model 1 (APTOS): {e}")

        # 2. Model 2: DRIVE Vessel Segmentation
        self.vessel_model = None
        vessel_path = os.path.join(weights_dir, "vessel_unet.pth")
        try:
            if not os.path.isfile(vessel_path):
                raise FileNotFoundError(f"Vessel checkpoint not found at: {vessel_path}")
            vessel_net = DriveUNet(in_ch=1, out_ch=1, base=32).to(self.device)
            vessel_net.load_state_dict(torch.load(vessel_path, map_location=self.device, weights_only=True))
            vessel_net.eval()
            self.vessel_model = vessel_net
            logger.info("Model 2 (DRIVE) successfully loaded into memory.")
        except Exception as e:
            logger.error(f"Failed to load Model 2 (DRIVE): {e}")

        # 3. Model 3: IDRiD Retinal Lesion Segmentation
        self.idrid_model = None
        idrid_path = os.path.join(weights_dir, "idrid_best_model.pth")
        if not os.path.isfile(idrid_path):
            idrid_path = os.path.join(weights_dir, "best_model.pth")
        try:
            if not os.path.isfile(idrid_path):
                raise FileNotFoundError(f"IDRiD checkpoint not found at: {idrid_path}")
            checkpoint = torch.load(idrid_path, map_location=self.device, weights_only=False)
            idrid_net = IdridUNet(n_channels=3, n_classes=4, base_c=32).to(self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            idrid_net.load_state_dict(state_dict)
            idrid_net.eval()
            self.idrid_model = idrid_net
            logger.info("Model 3 (IDRiD) successfully loaded into memory.")
        except Exception as e:
            logger.error(f"Failed to load Model 3 (IDRiD): {e}")

        self._initialized = True

    @property
    def is_aptos_loaded(self) -> bool:
        return self.aptos_model is not None

    @property
    def is_vessel_loaded(self) -> bool:
        return self.vessel_model is not None

    @property
    def is_idrid_loaded(self) -> bool:
        return self.idrid_model is not None

registry = ModelRegistry()
