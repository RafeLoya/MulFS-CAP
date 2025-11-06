import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
import time

import model.model as model
import utils.utils as utils
import args


# =============================================================================
# SETUP
# =============================================================================

device_id = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")

# =============================================================================
# MODEL INIT
# =============================================================================

with torch.no_grad():
    base = model.base()
    vis_MFE = model.FeatureExtractor()
    ir_MFE = model.FeatureExtractor()
    fusion_decoder = model.Decoder()
    PAFE = model.FeatureExtractor()
    decoder = model.Decoder()
    MN_vis = model.Enhance()
    MN_ir = model.Enhance()
    VISDP = model.DictionaryRepresentationModule()
    IRDP = model.DictionaryRepresentationModule()
    ImageDeformation = model.ImageTransform().cpu()
    MHCSA_vis = model.MHCSAB()
    MHCSA_ir = model.MHCSAB()
    fusion_module = model.FusionMoudle()

pretrain_dir = r"./pretrain"
checkpoints = torch.load(os.path.join(pretrain_dir, "ckpts.pth"))

utils.load_state_dir(base, checkpoints['bfe'], device)
utils.load_state_dir(vis_MFE, checkpoints['vis_mfe'], device)
utils.load_state_dir(ir_MFE, checkpoints['ir_mfe'], device)
utils.load_state_dir(PAFE, checkpoints['pafe'], device)
utils.load_state_dir(VISDP, checkpoints['vis_dgfp'], device)
utils.load_state_dir(IRDP, checkpoints['ir_dgfp'], device)
utils.load_state_dir(MN_vis, checkpoints['mn_vis'], device)
utils.load_state_dir(MN_ir, checkpoints['mn_ir'], device)
utils.load_state_dir(MHCSA_vis, checkpoints['mhcsab_vis'], device)
utils.load_state_dir(MHCSA_ir, checkpoints['mhcsab_ir'], device)
utils.load_state_dir(fusion_module, checkpoints['fusion_block'], device)

base.eval()
vis_MFE.eval()
ir_MFE.eval()
PAFE.eval()
VISDP.eval()
IRDP.eval()
MN_vis.eval()
MN_ir.eval()
MHCSA_vis.eval()
MHCSA_ir.eval()
fusion_module.eval()

# =============================================================================
# PROCESSING
# =============================================================================

class RealTimeFusion:
    def __init__(self, vis_source, ir_source, use_fp16=True):
        self.cap_vis = cv2.VideoCapture(vis_source)
        self.cap_ir = cv2.VideoCapture(ir_source)
        self.use_fp16 = use_fp16

        if not self.cap_vis.isOpened() or not self.cap_ir.isOpened():
            raise ValueError("[ERR] could not open video source(s)")

        self.fps = self.cap_ir.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] video FPS: {self.fps}")

    def preprocess_frame(self, frame):
        """OpenCV frame -> PyTorch tensor"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = frame.shape

        # tensor [1, 1, H, W]
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor, (h, w)

    def postprocess_tensor(self, tensor, original_size=None):
        """PyTorch tensor -> OpenCV frame"""
        frame = tensor.squeeze().cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)

        if original_size is not None:
            h, w = original_size
            frame = cv2.resize(frame, (w, h))

        return frame

    def process_frame_pair(self, vis_tensor, ir_tensor):
        """MulFS-CAP processing"""

        # arbitrary resolution
        _, _, h, w = vis_tensor.size()
        original_h, original_w = h, w

        if h % 16 != 0 or w % 16 != 0:
            new_h = (h // 16) * 16
            new_w = (w // 16) * 16
            ir_tensor = F.interpolate(ir_tensor, (new_h, new_w), mode="bilinear", align_corners=False)
            vis_tensor = F.interpolate(vis_tensor, (new_h, new_w), mode="nearest", align_corners=False)

        with torch.no_grad():
            # image deformation
            _, ir_d, _, _, _ = ImageDeformation(vis_tensor, ir_tensor)

            # feature extraction
            vis_1 = base(vis_tensor)
            ir_d_1 = base(ir_d)
            vis_fe = vis_MFE(vis_1)
            ir_d_fe = ir_MFE(ir_d_1)
            vis_f = PAFE(vis_1)
            ir_d_f = PAFE(ir_d_1)

            # feature enhancement
            vis_e_f = MN_vis(vis_f)
            ir_d_e_f = MN_ir(ir_d_f)
            VISDP_vis_f, _ = VISDP(vis_e_f)
            IRDP_ir_d_f, _ = IRDP(ir_d_e_f)

            # cross-modality alignment
            fixed_DP = VISDP_vis_f
            moving_DP = IRDP_ir_d_f
            moving_DP_lw = model.df_window_partition(moving_DP, args.args.large_w_size, args.args.small_w_size)
            # TODO Why is it passing two `small_w_size` and not large & small in original?
            fixed_DP_sw = model.window_partition(fixed_DP, args.args.small_w_size, args.args.small_w_size)
            corresoondence_matrices = model.CMAP(fixed_DP_sw, moving_DP_lw, MHCSA_vis, MHCSA_ir, True)

            # feature reorganization & fusion
            ir_d_f_sample = model.feature_reorganization(corresoondence_matrices, ir_d_fe)
            fusion_image_sample = fusion_module(vis_fe, ir_d_f_sample)

            # resize back
            if original_h % 16 != 0 or original_w % 16 != 0:
                fusion_image_sample = F.interpolate(fusion_image_sample, (original_h, original_w), mode="bilinear", align_corners=False)

        return fusion_image_sample

    def run(self, display=True, save_video=None):
        """Main processing loop"""
        frame_times = []
        frame_count = 0

        # video writer
        writer = None
        fourcc = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # init after first frame to get dimensions

        print("[INFO] starting real-time processing, press 'q' to quit")

        try:
            while True:
                start_time = time.time()

                # read frames
                ret_vis, frame_vis = self.cap_vis.read()
                ret_ir ,frame_ir = self.cap_ir.read()

                if not ret_vis or not ret_ir:
                    print("[INFO] end of video")
                    break

                # preprocess
                vis_tensor, vis_size = self.preprocess_frame(frame_vis)
                ir_tensor, ir_size = self.preprocess_frame(frame_ir)

                vis_tensor = vis_tensor.to(device)
                ir_tensor = ir_tensor.to(device)

                # if enabled, proc w/ FP16
                if self.use_fp16:
                    with autocast():
                        fused_tensor = self.process_frame_pair(vis_tensor, ir_tensor)
                else:
                    fused_tensor = self.process_frame_pair(vis_tensor, ir_tensor)

                # postprocess
                fused_frame = self.postprocess_tensor(fused_tensor)

                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                frame_count += 1

                # display
                if display:
                    if len(frame_times) >= 30:
                        avg_fps = 1 / np.mean(frame_times[-30:])
                    else:
                        avg_fps = 1 / len(frame_times)

                    display_frame = cv2.cvtColor(fused_frame, cv2.COLOR_GRAY2BGR)
                    cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                                cv2.FONT_HERHSEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Fused Output', display_frame)

                    # show inputs
                    cv2.imshow('Visible Input', frame_vis)
                    cv2.imshow('Infrared Input', frame_ir)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] quit requested")
                        break

                # save to video
                if save_video:
                    if writer is None:
                        h, w = fused_frame.shape
                        writer = cv2.VideoWriter(save_video, fourcc, self.fps, (w, h), False)
                    writer.write(fused_frame)

                # output progress
                if frame_count % 30 == 0:
                    avg_fps = 1 / np.mean(frame_times[-30:])
                    print(f"Frame {frame_count}: {avg_fps:.2f} FPS")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")

        finally:
            # cleanup
            self.cap_vis.release()
            self.cap_ir.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            if frame_times:
                avg_time = np.mean(frame_times)
                print(f"[INFO] Processing complete!")
                print(f"  total frames: {frame_count}")
                print(f"  average time per frame: {avg_time * 1000:.1f}ms")
                print(f"  average fps: {1/avg_time:.2f}")

if __name__ == "__main__":
    VIS_SOURCE = "./data/test_videos/visible.mp4"
    IR_SOURCE = "./data/test_videos/infrared.mp4"
    USE_FP16 = True
    DISPLAY = True
    SAVE_OUTPUT = "./results/fusion.mp4"
    # SAVE_OUTPUT = None

    fusion = RealTimeFusion(VIS_SOURCE, IR_SOURCE, use_fp16=USE_FP16)
    fusion.run(display=DISPLAY, save_video=SAVE_OUTPUT)