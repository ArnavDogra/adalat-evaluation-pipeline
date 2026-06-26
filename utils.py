import logging
import platform
import sys
from pathlib import Path

def setup_logging(output_dir: Path):
    log_file = output_dir / 'pipeline.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def validate_environment(logger):
    import torch
    import transformers
    import ctranslate2
    try:
        import faster_whisper
        fw_version = faster_whisper.__version__
    except:
        fw_version = "Not installed"
    
    logger.info("=== Environment Validation ===")
    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"Torch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    logger.info(f"Transformers Version: {transformers.__version__}")
    logger.info(f"CTranslate2 Version: {ctranslate2.__version__}")
    logger.info(f"Faster-Whisper Version: {fw_version}")
    logger.info("==============================")
    
    return 'cuda' if cuda_available else 'cpu'
