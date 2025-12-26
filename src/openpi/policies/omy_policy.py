# src/openpi/policies/omy_policy.py
import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # CHW -> HWC
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class OmyInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # robot client가 보내는 키 기준:
        # data["state"] : (7,)
        # data["images"]["cam_third_color"], ["cam_wrist_color"], ["cam_top"] : CHW uint8 or HWC
        images_in = data["images"]

        base = _parse_image(images_in["cam_third_color"])
        wrist = _parse_image(images_in["cam_wrist_color"])
        top = _parse_image(images_in["cam_top"]) if "cam_top" in images_in else np.zeros_like(base)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                imgs = (base, wrist, top)
                masks = (np.True_, np.True_, np.True_ if "cam_top" in images_in else np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                imgs = (base, top, wrist)
                # FAST는 padding mask를 안 끄는 패턴을 Droid가 사용【openpi.txt†L13233-L13238】
                masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": dict(zip(names, imgs, strict=True)),
            "image_mask": dict(zip(names, masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"].decode("utf-8") if isinstance(data["prompt"], bytes) else data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OmyOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # OMY는 7차원(6 arm + gripper)만 사용
        return {"actions": np.asarray(data["actions"][:, :7])}
