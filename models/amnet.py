"""AMNet: Adaptive Multi-modal Network for low-light video enhancement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from .blocks import ChannelLayerNorm
from .encoder_decoder import (RetinexImageEncoder, RetinexDecoder,
                              LightweightEventEncoder, LightweightIREncoder)
from .retinex import MultiModalIlluminationEstimator, SingleScaleSNRAwareFusion, SNRMapGenerator
from .adapter import S2DGTranslator
from .convlstm import ConvLSTM
from utils.metrics import CharbonnierLoss, ssim_loss


class AMNet(nn.Module):
    """Adaptive Multi-modal Network for low-light video enhancement."""

    def __init__(
        self,
        in_ch=3,
        encoder_channels=[64, 128, 256, 256],
        latent_dim=256,
        convlstm_hidden_ch=256,
        convlstm_layers=1,
        event_enable=True,
        ir_enable=True,
        event_in_ch=10,
        event_base_ch=32,
        ir_base_ch=32,
        encoder_num_blocks=[1, 1, 2, 2],
        decoder_num_blocks=[2, 2, 1, 1],
        use_multimodal_illumination=True,
        snr_factor=1.0,
        snr_threshold=0.5,
        snr_fusion_depth=1,
        use_snr_guided_fusion=True,
        use_checkpoint=False,
        checkpoint_encoder=True,
        checkpoint_decoder=False,
        checkpoint_aux_encoder=True,
        checkpoint_fusion=True,
        **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.event_enable = event_enable
        self.ir_enable = ir_enable
        self.use_multimodal_illumination = use_multimodal_illumination
        self.use_snr_guided_fusion = use_snr_guided_fusion

        self.use_checkpoint = use_checkpoint
        self.checkpoint_encoder = use_checkpoint and checkpoint_encoder
        self.checkpoint_decoder = use_checkpoint and checkpoint_decoder
        self.checkpoint_aux_encoder = use_checkpoint and checkpoint_aux_encoder
        self.checkpoint_fusion = use_checkpoint and checkpoint_fusion

        self.encoder = RetinexImageEncoder(
            in_ch=in_ch, encoder_channels=encoder_channels,
            num_blocks=encoder_num_blocks, dim_head=32
        )
        C_img = self.encoder.out_ch
        self.ln_img = ChannelLayerNorm(C_img)
        self.proj_img = nn.Conv2d(C_img, latent_dim, 1)

        if use_multimodal_illumination:
            self.mm_illu_estimator = MultiModalIlluminationEstimator(
                n_fea_middle=encoder_channels[0], rgb_ch=in_ch,
                event_ch=event_in_ch if event_enable else 0,
                ir_ch=1 if ir_enable else 0, n_fea_out=in_ch
            )

        self.snr_generator = SNRMapGenerator(snr_factor=snr_factor)

        if event_enable:
            self.event_encoder = LightweightEventEncoder(
                in_ch=event_in_ch, base_ch=event_base_ch,
                latent_dim=latent_dim, use_checkpoint=self.checkpoint_aux_encoder
            )
            self.hallucinate_event = S2DGTranslator(
                encoder_channels=encoder_channels, latent_dim=latent_dim
            )
            if use_snr_guided_fusion:
                self.snr_fusion = SingleScaleSNRAwareFusion(
                    latent_dim=latent_dim, snr_threshold=snr_threshold,
                    depth=snr_fusion_depth, use_event=event_enable,
                    use_ir=ir_enable, use_checkpoint=self.checkpoint_fusion
                )

        if ir_enable:
            self.ir_encoder = LightweightIREncoder(
                in_ch=1, base_ch=ir_base_ch,
                latent_dim=latent_dim, use_checkpoint=self.checkpoint_aux_encoder
            )
            self.hallucinate_ir = S2DGTranslator(
                encoder_channels=encoder_channels, latent_dim=latent_dim
            )

        self.convlstm = ConvLSTM(
            input_dim=latent_dim, hidden_dim=convlstm_hidden_ch,
            kernel_size=3, num_layers=convlstm_layers,
        )
        self.post_lstm_proj = nn.Conv2d(convlstm_hidden_ch, latent_dim, 1)

        self.decoder = RetinexDecoder(
            encoder_channels=encoder_channels, latent_dim=latent_dim,
            out_ch=in_ch, num_blocks=decoder_num_blocks, dim_head=32
        )

        self.criterion_pixel = CharbonnierLoss(eps=1e-3)

        self._stream_hidden = None
        self._stream_spatial = None

    def extract_rgb_feats(self, x):
        """Pre-compute RGB encoder features for all frames."""
        B, T, C, H, W = x.shape
        feats_list = []
        for t in range(T):
            enc_feats, enc_illu_feas = self.encoder(x[:, t])
            feats_list.append((enc_feats, enc_illu_feas))
        return feats_list

    def _run_encoder_tuple(self, x):
        enc_feats, enc_illu_feas = self.encoder(x)
        return tuple(enc_feats), tuple(enc_illu_feas)

    def _generate_blur_image(self, img):
        blur_kernel = torch.ones(1, 1, 5, 5, device=img.device) / 25.0
        blur_kernel = blur_kernel.repeat(img.shape[1], 1, 1, 1)
        return F.conv2d(img, blur_kernel, padding=2, groups=img.shape[1])

    def forward(self, x, y=None, event=None, ir=None,
                precomputed_feats=None, detach_aux=False, loss_cfg=None,
                event_use_real=None, ir_use_real=None):
        """Forward pass. Training (y given): returns loss. Inference: returns enhanced frames."""
        if y is not None:
            return self._forward_train(
                x, y, event=event, ir=ir, loss_cfg=loss_cfg,
                precomputed_feats=precomputed_feats
            )
        return self._forward_impl(
            x, event=event, ir=ir, precomputed_feats=precomputed_feats,
            detach_aux=detach_aux, event_use_real=event_use_real,
            ir_use_real=ir_use_real,
        )

    def _forward_impl(self, x, event=None, ir=None,
                      precomputed_feats=None, detach_aux=False,
                      event_use_real=None, ir_use_real=None):
        """Inference forward: process video frame-by-frame with ConvLSTM."""
        B, T, C, H, W = x.shape
        device = x.device

        H8, W8 = H // 8, W // 8
        hidden = self.convlstm.init_hidden(B, (H8, W8), device=device)
        outputs = []
        aux_record = {"z_rec_event": [], "z_real_event": [],
                      "z_rec_ir": [], "z_real_ir": []}

        for t in range(T):
            frame = x[:, t]

            if precomputed_feats is not None:
                enc_feats, enc_illu_feas = precomputed_feats[t]
                if isinstance(enc_feats, tuple):
                    enc_feats = list(enc_feats)
                if isinstance(enc_illu_feas, tuple):
                    enc_illu_feas = list(enc_illu_feas)
            else:
                if self.training and self.checkpoint_encoder:
                    enc_feats_t, enc_illu_feas_t = ckpt(
                        self._run_encoder_tuple, frame, use_reentrant=False
                    )
                    enc_feats = list(enc_feats_t)
                    enc_illu_feas = list(enc_illu_feas_t)
                else:
                    enc_feats, enc_illu_feas = self.encoder(frame)

            feat_rgb_last = enc_feats[-1]
            z_img = self.proj_img(self.ln_img(feat_rgb_last))

            enc_feats_for_aux = [f.detach() for f in enc_feats] if detach_aux else enc_feats

            z_event_target = None
            if self.event_enable:
                z_event_rec = self.hallucinate_event(
                    enc_feats_for_aux,
                    use_checkpoint=self.use_checkpoint and self.training
                )
                aux_record["z_rec_event"].append(z_event_rec)

                z_event_real = None
                if event is not None:
                    z_event_real = self.event_encoder(event[:, t])
                    aux_record["z_real_event"].append(z_event_real)

                if event_use_real is True and z_event_real is not None:
                    z_event_target = z_event_real
                elif event_use_real is False:
                    z_event_target = z_event_rec
                else:
                    z_event_target = z_event_real if z_event_real is not None else z_event_rec
            else:
                z_event_target = torch.zeros_like(z_img)
                aux_record["z_rec_event"].append(None)
                aux_record["z_real_event"].append(None)

            z_ir_target = None
            if self.ir_enable:
                z_ir_rec = self.hallucinate_ir(
                    enc_feats_for_aux,
                    use_checkpoint=self.use_checkpoint and self.training
                )
                aux_record["z_rec_ir"].append(z_ir_rec)

                z_ir_real = None
                if ir is not None:
                    z_ir_real = self.ir_encoder(ir[:, t])
                    aux_record["z_real_ir"].append(z_ir_real)

                if ir_use_real is True and z_ir_real is not None:
                    z_ir_target = z_ir_real
                elif ir_use_real is False:
                    z_ir_target = z_ir_rec
                else:
                    z_ir_target = z_ir_real if z_ir_real is not None else z_ir_rec
            else:
                z_ir_target = torch.zeros_like(z_img)
                aux_record["z_rec_ir"].append(None)
                aux_record["z_real_ir"].append(None)

            use_real_for_snr_event = (event_use_real is True) and (event is not None)
            use_real_for_snr_ir = (ir_use_real is True) and (ir is not None)

            if hasattr(self, 'mm_illu_estimator'):
                _, pred_illu = self.mm_illu_estimator(
                    frame,
                    event=event[:, t] if use_real_for_snr_event else None,
                    ir=ir[:, t] if use_real_for_snr_ir else None
                )
            else:
                _, pred_illu = self.encoder.illu_estimator(frame)

            enhanced_mid = frame * pred_illu + frame
            enhanced_blur = self._generate_blur_image(enhanced_mid)
            snr_map_full = self.snr_generator(enhanced_mid, enhanced_blur)
            snr_map = F.adaptive_avg_pool2d(snr_map_full, (H8, W8))

            z_fused = self.snr_fusion(
                rgb_feat=z_img, snr_map=snr_map, att_feat=z_img,
                event_feat=z_event_target,
                ir_feat=z_ir_target if self.ir_enable else None
            )

            h, hidden = self.convlstm.forward_step(z_fused, hidden)
            h = self.post_lstm_proj(h)

            if self.checkpoint_decoder and self.training:
                def run_decoder(h, enc_feats, enc_illu_feas):
                    return self.decoder(h, enc_feats, enc_illu_feas)
                residual = ckpt(run_decoder, h, enc_feats, enc_illu_feas, use_reentrant=False)
            else:
                residual = self.decoder(h, enc_feats, enc_illu_feas)

            pred_frame = torch.clamp(frame + residual, 0.0, 1.0)
            outputs.append(pred_frame)

        out_stack = torch.stack(outputs, dim=1)

        def stack_aux(key):
            if len(aux_record[key]) > 0 and aux_record[key][0] is not None:
                return torch.stack(aux_record[key], dim=1)
            return None

        loss_data = {
            "z_rec_event": stack_aux("z_rec_event"),
            "z_real_event": stack_aux("z_real_event"),
            "z_rec_ir": stack_aux("z_rec_ir"),
            "z_real_ir": stack_aux("z_real_ir"),
        }
        return out_stack, loss_data

    def _forward_train(self, x, y, event=None, ir=None, loss_cfg=None, precomputed_feats=None):
        """Training forward: compute loss over multiple modality combinations."""
        feats = precomputed_feats if precomputed_feats is not None else self.extract_rgb_feats(x)

        cfg = loss_cfg or {}
        lam_rob = cfg.get("lambda_robust", 1.0)
        lam_real = cfg.get("lambda_real", 0.5)
        lam_ssim = cfg.get("lambda_ssim", 0.2)
        lam_de = cfg.get("lambda_distill_event", 0.1)
        lam_di = cfg.get("lambda_distill_ir", 0.5)
        dist_f = cfg.get("distill_factor", 0.0)
        use_real_event_train = cfg.get("train_use_real_event", True) and (event is not None)
        use_real_ir_train = cfg.get("train_use_real_ir", True) and (ir is not None)

        combos = []
        ev_opts = [True, False] if use_real_event_train else [False]
        ir_opts = [True, False] if use_real_ir_train else [False]
        for ev_flag in ev_opts:
            for ir_flag in ir_opts:
                combos.append((ev_flag, ir_flag))
        if (False, False) not in combos:
            combos.append((False, False))
        combos.sort(key=lambda c: (c[0], c[1]), reverse=True)

        total_loss_accum = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        loss_distill_event = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        loss_distill_ir = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        combo_losses = {}
        teacher_feats = None
        teacher_loss_val = 1e9
        robust_loss_val = 0.0

        for idx_c, (ev_real_flag, ir_real_flag) in enumerate(combos):
            is_teacher = (idx_c == 0)
            is_student = (not ev_real_flag) and (not ir_real_flag)

            pred_mm, loss_data_curr = self._forward_impl(
                x, event=event, ir=ir, precomputed_feats=feats, detach_aux=False,
                event_use_real=ev_real_flag if event is not None else None,
                ir_use_real=ir_real_flag if ir is not None else None,
            )
            pred_mm = torch.clamp(pred_mm, 0.0, 1.0)

            loss_pixel = self.criterion_pixel(pred_mm, y)
            loss_comb = loss_pixel + lam_ssim * ssim_loss(pred_mm, y)

            weight = lam_real if is_teacher else lam_rob
            total_loss_accum = total_loss_accum + (weight * loss_comb)

            combo_name = f"loss_{'real' if ev_real_flag else 'fake'}E_{'real' if ir_real_flag else 'fake'}IR"
            combo_losses[combo_name] = loss_comb

            if is_teacher:
                teacher_feats = loss_data_curr
                teacher_loss_val = loss_comb.detach()
            if is_student:
                robust_loss_val = loss_comb.detach()

        total_loss = total_loss_accum / float(len(combos))
        weighted_distill = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        use_distill = (dist_f > 0) and (teacher_feats is not None) and (len(combos) > 1)

        if use_distill:
            def cosine_similarity_loss(feat1, feat2):
                if feat1 is None or feat2 is None:
                    device = feat1.device if feat1 is not None else feat2.device
                    return torch.tensor(0.0, device=device)
                B, T, C, H, W = feat1.shape
                f1 = F.normalize(feat1.view(B * T, -1), p=2, dim=1)
                f2 = F.normalize(feat2.view(B * T, -1), p=2, dim=1)
                return (1 - (f1 * f2).sum(dim=1).clamp(-1, 1)).mean()

            student_feats = loss_data_curr

            if (event is not None) and (teacher_feats.get("z_real_event") is not None):
                loss_distill_event = cosine_similarity_loss(
                    student_feats["z_rec_event"],
                    teacher_feats["z_real_event"].detach()
                )

            if (ir is not None) and (teacher_feats.get("z_real_ir") is not None):
                loss_distill_ir = cosine_similarity_loss(
                    student_feats["z_rec_ir"],
                    teacher_feats["z_real_ir"].detach()
                )

            weighted_distill = dist_f * (lam_de * loss_distill_event + lam_di * loss_distill_ir)
            total_loss = total_loss + weighted_distill

        loss_dict = {
            "loss_fakeE_fakeIR": float(robust_loss_val),
            "loss_distill_weighted": float(weighted_distill.item()),
            "loss_distill_event": float(loss_distill_event.item()),
            "loss_distill_ir": float(loss_distill_ir.item()),
            **{k: float(v.item()) for k, v in combo_losses.items()}
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def inference_video(self, x, event=None, ir=None,
                        event_use_real=None, ir_use_real=None):
        """Offline video inference. Input/output: [T, C, H, W]."""
        self.eval()
        x_b = x.unsqueeze(0)
        event_b = event.unsqueeze(0) if event is not None else None
        ir_b = ir.unsqueeze(0) if ir is not None else None
        res, _ = self.forward(x_b, event=event_b, ir=ir_b,
                              event_use_real=event_use_real, ir_use_real=ir_use_real)
        return res.squeeze(0)

    @torch.no_grad()
    def start_stream(self, batch_size, image_size, device=None):
        """Initialize streaming inference state."""
        H, W = image_size
        H8, W8 = H // 8, W // 8
        device = device or next(self.parameters()).device
        self._stream_hidden = self.convlstm.init_hidden(batch_size, (H8, W8), device=device)
        self._stream_spatial = (H, W)

    @torch.no_grad()
    def stream_step(self, frame, event=None, ir=None,
                    event_use_real=None, ir_use_real=None):
        """Single-frame streaming inference. Returns enhanced frame [B, 3, H, W]."""
        assert self._stream_hidden is not None, "Call start_stream first"
        B, _, H, W = frame.shape
        assert (H, W) == self._stream_spatial
        H8, W8 = H // 8, W // 8

        enc_feats, enc_illu_feas = self.encoder(frame)
        feat_rgb_last = enc_feats[-1]
        z_img = self.proj_img(self.ln_img(feat_rgb_last))

        z_event_target = None
        if self.event_enable:
            z_event_rec = self.hallucinate_event(enc_feats, use_checkpoint=False)
            z_event_real = None
            if event is not None:
                z_event_real = self.event_encoder(event)
            if event_use_real is True and z_event_real is not None:
                z_event_target = z_event_real
            elif event_use_real is False:
                z_event_target = z_event_rec
            else:
                z_event_target = z_event_real if z_event_real is not None else z_event_rec
        else:
            z_event_target = torch.zeros_like(z_img)

        z_ir_target = None
        if self.ir_enable:
            z_ir_rec = self.hallucinate_ir(enc_feats, use_checkpoint=False)
            z_ir_real = None
            if ir is not None:
                z_ir_real = self.ir_encoder(ir)
            if ir_use_real is True and z_ir_real is not None:
                z_ir_target = z_ir_real
            elif ir_use_real is False:
                z_ir_target = z_ir_rec
            else:
                z_ir_target = z_ir_real if z_ir_real is not None else z_ir_rec
        else:
            z_ir_target = torch.zeros_like(z_img)

        use_real_for_snr_event = (event_use_real is True) and (event is not None)
        use_real_for_snr_ir = (ir_use_real is True) and (ir is not None)

        if hasattr(self, 'mm_illu_estimator'):
            _, pred_illu = self.mm_illu_estimator(
                frame,
                event=event if use_real_for_snr_event else None,
                ir=ir if use_real_for_snr_ir else None
            )
        else:
            _, pred_illu = self.encoder.illu_estimator(frame)

        enhanced_mid = frame * pred_illu + frame
        enhanced_blur = self._generate_blur_image(enhanced_mid)
        snr_map_full = self.snr_generator(enhanced_mid, enhanced_blur)
        snr_map = F.adaptive_avg_pool2d(snr_map_full, (H8, W8))

        z_fused = self.snr_fusion(
            rgb_feat=z_img, snr_map=snr_map, att_feat=z_img,
            event_feat=z_event_target,
            ir_feat=z_ir_target if self.ir_enable else None
        )

        h, self._stream_hidden = self.convlstm.forward_step(z_fused, self._stream_hidden)
        h = self.post_lstm_proj(h)
        residual = self.decoder(h, enc_feats, enc_illu_feas)
        return torch.clamp(frame + residual, 0.0, 1.0)

    @torch.no_grad()
    def end_stream(self):
        """Clear streaming state."""
        self._stream_hidden = None
        self._stream_spatial = None
