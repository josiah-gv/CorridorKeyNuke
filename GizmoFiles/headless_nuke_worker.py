import os
import sys
import argparse
import logging
import warnings
import traceback

# Enable OpenEXR Support (Must be before cv2 import)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_corridor_key_engine(device='cuda'):
    try:
        from CorridorKeyModule.inference_engine import CorridorKeyEngine
        import glob
        
        # Auto-detect checkpoint
        ckpt_dir = os.path.join(BASE_DIR, "CorridorKeyModule", "checkpoints")
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
        
        if len(ckpt_files) == 0:
            raise FileNotFoundError(f"No .pth checkpoint found in {ckpt_dir}")
        elif len(ckpt_files) > 1:
            logger.warning(f"Multiple checkpoints found in {ckpt_dir}. Using the first one: {os.path.basename(ckpt_files[0])}")
            
        ckpt_path = ckpt_files[0]
        logger.info(f"Using checkpoint: {os.path.basename(ckpt_path)}")
        
        return CorridorKeyEngine(checkpoint_path=ckpt_path, device=device, img_size=2048)
    except Exception as e:
        logger.error(f"Failed to initialize CorridorKey Engine: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="CorridorKey Headless Nuke Worker")
    parser.add_argument("--plate_dir", required=True, help="Path to input Plate sequence")
    parser.add_argument("--alpha_dir", required=True, help="Path to input AlphaHint sequence")
    parser.add_argument("--output_dir", required=True, help="Path to write Outputs (will create FG, Matte, Processed subdirs)")
    parser.add_argument("--start_frame", type=int, required=True, help="Start frame number")
    parser.add_argument("--end_frame", type=int, required=True, help="End frame number")
    
    # Optional settings matching the UI
    parser.add_argument("--gamma", type=str, choices=['linear', 'srgb'], default='srgb', help="Input Gamma Space")
    parser.add_argument("--despill", type=float, default=1.0, help="Despill Strength (0.0 to 1.0)")
    parser.add_argument("--auto_despeckle", action="store_true", help="Enable Auto-Despeckle")
    parser.add_argument("--despeckle_size", type=int, default=400, help="Despeckle Size Threshold")
    parser.add_argument("--refiner_scale", type=float, default=1.0, help="Refiner Strength")
    
    args = parser.parse_args()
    
    # 1. Setup Output Dirs
    fg_dir = os.path.join(args.output_dir, "FG")
    matte_dir = os.path.join(args.output_dir, "Matte")
    proc_dir = os.path.join(args.output_dir, "Processed")
    comp_dir = os.path.join(args.output_dir, "Comp")
    
    for d in [fg_dir, matte_dir, proc_dir, comp_dir]:
        os.makedirs(d, exist_ok=True)
        
    exr_flags = [
        cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
        cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
    ]
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = get_corridor_key_engine(device=device)
    
    input_is_linear = (args.gamma.lower() == 'linear')
    
    # Find files based on frame range. We assume Nuke exported them matching 'start_frame' to 'end_frame', padded.
    # Nuke will likely export as `input.####.exr` or similar. Let's just find files that match.
    import glob
    plate_files = sorted(glob.glob(os.path.join(args.plate_dir, "*.*")))
    alpha_files = sorted(glob.glob(os.path.join(args.alpha_dir, "*.*")))
    
    if len(plate_files) == 0:
        logger.error(f"No plate files found in {args.plate_dir}")
        sys.exit(1)
    if len(alpha_files) == 0:
        logger.error(f"No alpha hint files found in {args.alpha_dir}")
        sys.exit(1)
        
    # Zip them together based on index. (In a single frame export, they will just be one file each).
    for i in range(min(len(plate_files), len(alpha_files))):
        p_file = plate_files[i]
        a_file = alpha_files[i]
        
        input_stem = os.path.basename(p_file).split('.')[0]
        logger.info(f"Processing: {os.path.basename(p_file)}")
        
        # 2. Read Plate
        is_exr = p_file.lower().endswith('.exr')
        img_srgb = None
        if is_exr:
            img_linear = cv2.imread(p_file, cv2.IMREAD_UNCHANGED)
            if img_linear is None: 
                logger.error(f"Failed to read EXR plate: {p_file}")
                continue
            # Nuke exports EXRs linearly. Convert to sRGB via OpenEXR rules if user didn't specify 'linear' mode.
            img_linear_rgb = cv2.cvtColor(img_linear, cv2.COLOR_BGR2RGB)
            img_srgb = np.maximum(img_linear_rgb, 0.0) # We always pass float 0-1 to engine
        else:
            img_bgr = cv2.imread(p_file)
            if img_bgr is None: 
                logger.error(f"Failed to read Plate: {p_file}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_srgb = img_rgb.astype(np.float32) / 255.0
            
        # 3. Read Alpha Hint
        mask_in = cv2.imread(a_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if mask_in is None:
            logger.error(f"Failed to read AlphaHint: {a_file}")
            continue
            
        mask_linear = None
        if mask_in.ndim == 3:
             if mask_in.shape[2] >= 3:
                 mask_linear = mask_in[:, :, 0] # Extract Red channel
             else:
                 mask_linear = mask_in
        else:
            mask_linear = mask_in
        
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0
        elif mask_linear.dtype == np.uint16:
            mask_linear = mask_linear.astype(np.float32) / 65535.0
        else:
            mask_linear = mask_linear.astype(np.float32)

        if mask_linear.shape[:2] != img_srgb.shape[:2]:
             mask_linear = cv2.resize(mask_linear, (img_srgb.shape[1], img_srgb.shape[0]), interpolation=cv2.INTER_LINEAR)
             
        # 4. Process
        try:
            res = engine.process_frame(
                img_srgb, 
                mask_linear, 
                input_is_linear=input_is_linear, 
                fg_is_straight=True, # Always use straight model output
                despill_strength=args.despill,
                auto_despeckle=args.auto_despeckle,
                despeckle_size=args.despeckle_size,
                refiner_scale=args.refiner_scale
            )
        except Exception as e:
            logger.error(f"Engine Exception: {e}")
            traceback.print_exc()
            continue
            
        pred_fg = res['fg'] # sRGB
        pred_alpha = res['alpha'] # Linear
        
        # 5. Save Outputs
        fg_bgr = cv2.cvtColor(pred_fg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(fg_dir, f"{input_stem}.exr"), fg_bgr, exr_flags)
        
        if pred_alpha.ndim == 3: pred_alpha = pred_alpha[:, :, 0]
        cv2.imwrite(os.path.join(matte_dir, f"{input_stem}.exr"), pred_alpha, exr_flags)
        
        comp_srgb = res['comp']
        comp_bgr = cv2.cvtColor((np.clip(comp_srgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(comp_dir, f"{input_stem}.png"), comp_bgr)

        if 'processed' in res:
            proc_rgba = res['processed']
            proc_bgra = cv2.cvtColor(proc_rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(os.path.join(proc_dir, f"{input_stem}.exr"), proc_bgra, exr_flags)
            
    logger.info("Headless Nuke Worker completed successfully.")

if __name__ == "__main__":
    main()
