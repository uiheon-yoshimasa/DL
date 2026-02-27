import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

# -----------------------------
# Grad-CAM core
# -----------------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; we want gradient wrt layer output
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        # full_backward_hook is safer for newer PyTorch
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """
        x: (1, 3, H, W)
        class_idx: target class index. If None, use top-1 predicted.
        returns: cam (H, W) in [0,1], class_idx, probs
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # (1, num_classes)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=False)

        # activations: (1, C, h, w), gradients: (1, C, h, w)
        grads = self.gradients
        acts = self.activations

        # global average pooling over spatial dims -> weights (1, C, 1, 1)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=False)  # (1, h, w)
        cam = F.relu(cam)

        cam = cam[0]
        cam = self._normalize_cam(cam)

        return cam.cpu().numpy(), class_idx, probs.detach().cpu().numpy()[0]


# -----------------------------
# Utility: overlay heatmap
# -----------------------------
def overlay_cam_on_image(bgr_img: np.ndarray, cam_01: np.ndarray, alpha: float = 0.4):
    h, w = bgr_img.shape[:2]
    cam_resized = cv2.resize(cam_01, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_img, 1.0 - alpha, heatmap, alpha, 0)
    return overlay, heatmap


def find_class_index_by_keyword(categories: list[str], keyword: str):
    key = keyword.lower()
    hits = [i for i, c in enumerate(categories) if key in c.lower()]
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--out", default="gradcam_overlay.png", help="output overlay image path")
    parser.add_argument("--heatmap_out", default="gradcam_heatmap.png", help="output heatmap image path")
    parser.add_argument("--alpha", type=float, default=0.4, help="overlay alpha (0-1)")
    parser.add_argument("--target", default="top1", help='target class: "top1" or keyword e.g. "bottle" / "water bottle"')
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # model + weights/meta
    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]

    model = models.resnet50(weights=weights).to(device)
    model.eval()

    # target layer: last conv block output
    target_layer = model.layer4[-1].conv3
    cam_extractor = GradCAM(model, target_layer)

    # preprocess
    preprocess = weights.transforms()

    pil = Image.open(args.image).convert("RGB")
    x = preprocess(pil).unsqueeze(0).to(device)

    # choose target class
    class_idx = None
    if args.target.lower() != "top1":
        hits = find_class_index_by_keyword(categories, args.target)
        if len(hits) == 0:
            print(f'[WARN] keyword "{args.target}" not found in ImageNet labels. Use top1 instead.')
            class_idx = None
        else:
            # pick the first match
            class_idx = hits[0]
            print(f'[INFO] keyword "{args.target}" matched: idx={class_idx}, label="{categories[class_idx]}"')

    # compute grad-cam
    cam_01, used_idx, probs = cam_extractor(x, class_idx=class_idx)
    pred_label = categories[used_idx]
    pred_prob = float(probs[used_idx])

    print(f"[RESULT] target idx={used_idx}, label={pred_label}, prob={pred_prob:.4f}")

    # load original image with OpenCV (BGR)
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to read image with OpenCV. Check path/format.")

    overlay, heatmap = overlay_cam_on_image(bgr, cam_01, alpha=args.alpha)

    cv2.imwrite(args.out, overlay)
    cv2.imwrite(args.heatmap_out, heatmap)

    print(f"[SAVED] overlay: {args.out}")
    print(f"[SAVED] heatmap: {args.heatmap_out}")

    cam_extractor.remove_hooks()


if __name__ == "__main__":
    main()