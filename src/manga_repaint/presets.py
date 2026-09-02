from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import JobMode


@dataclass(frozen=True, slots=True)
class PromptPreset:
    id: str
    name: str
    description: str
    prompt: str
    best_for: str = "通用漫画页面"
    changes: str = "颜色与光影"
    tradeoff: str = "效果会受到原图和模型能力影响"
    speed: str = "不改变推理速度"
    memory: str = "不额外增加显存"

    def to_json_dict(self) -> dict[str, str]:
        return asdict(self)


COLOR_PRESETS = {
    preset.id: preset
    for preset in (
        PromptPreset(
            "natural",
            "自然原作色",
            "肤色和室内光线自然，适合大多数剧情漫画",
            "natural skin tones, believable material colors, restrained saturation, "
            "consistent character colors and realistic local lighting",
            "剧情、日常与室内页面",
            "肤色、材质色和环境光",
            "风格表现相对克制",
        ),
        PromptPreset(
            "warm_cinematic",
            "暖调电影感",
            "暖肤色、木质色和柔和高光，适合室内与情感场景",
            "warm cinematic palette, amber practical lighting, warm skin tones, "
            "deep navy shadows and controlled highlights",
            "室内、情感与电影化场景",
            "整体色温、阴影和高光",
            "冷色场景可能偏暖",
        ),
        PromptPreset(
            "cool_night",
            "冷调夜景",
            "蓝紫阴影和低饱和环境光，适合悬疑与夜间场景",
            "cool blue and violet palette, low saturation ambient light, "
            "subtle warm skin contrast and deep clean shadows",
            "夜景、悬疑与低照度页面",
            "环境光和暗部色相",
            "白天场景可能显得偏冷",
        ),
        PromptPreset(
            "vivid_anime",
            "鲜明动画色",
            "颜色更鲜明，角色与背景层次清楚",
            "vivid anime palette, clean color separation, saturated focal colors, "
            "consistent skin and clothing colors without neon clipping",
            "动画感和角色展示页面",
            "饱和度与色块分离",
            "写实材质会被弱化",
        ),
        PromptPreset(
            "pastel",
            "柔和粉彩",
            "低对比柔和色彩，适合日常和轻喜剧场景",
            "soft pastel palette, gentle contrast, airy highlights, "
            "muted backgrounds and delicate natural skin tones",
            "日常、轻喜剧与柔和场景",
            "对比度和背景饱和度",
            "动作场景冲击力较弱",
        ),
        PromptPreset(
            "retro_print",
            "复古印刷色",
            "有限色盘和旧漫画印刷质感",
            "limited retro print palette, slightly faded inks, warm paper color, "
            "controlled registration texture and restrained saturation",
            "怀旧、印刷与限定色风格",
            "色盘、纸色和印刷质感",
            "颜色准确度会主动降低",
        ),
    )
}


STYLE_PRESETS = {
    preset.id: preset
    for preset in (
        PromptPreset(
            "original_ink",
            "原作线稿上色",
            "只改变颜色与光影，最大限度保留原始画法",
            "keep the original manga drawing style and original ink language; "
            "change only color, lighting and material rendering",
            "细节保护优先的上色",
            "颜色、光影与材质",
            "画风变化最小",
        ),
        PromptPreset(
            "modern_anime",
            "现代动画赛璐璐",
            "清晰色块和受控阴影，人物辨识度高",
            "modern high-end anime cel rendering, clean controlled shadow shapes, "
            "precise facial features and crisp material separation",
            "角色清晰、赛璐璐与动画页面",
            "阴影形状和材质边界",
            "网点和铅笔质感会减弱",
        ),
        PromptPreset(
            "soft_painted",
            "柔和厚涂",
            "更丰富的肤色和材质过渡，保留漫画构图",
            "soft painted illustration rendering, nuanced skin and cloth materials, "
            "subtle brush texture while preserving exact manga composition",
            "封面、插画与情绪页面",
            "肤色过渡、布料和笔触",
            "严格线稿感会降低",
        ),
        PromptPreset(
            "watercolor",
            "水彩漫画",
            "透明水彩铺色与轻柔纸张质感",
            "transparent watercolor manga rendering, delicate pigment variation, "
            "soft paper texture and restrained edges",
            "抒情、日常与轻柔页面",
            "透明色层和纸张质感",
            "硬边和高对比会变弱",
        ),
        PromptPreset(
            "dramatic_noir",
            "戏剧黑色电影",
            "保留大量黑白关系，以少量强调色塑造气氛",
            "dramatic noir color rendering, preserve strong black and white design, "
            "selective accent colors and cinematic rim light",
            "悬疑、动作与高反差页面",
            "黑白比例、强调色和轮廓光",
            "不适合需要丰富全彩的页面",
        ),
    )
}


# These controls make the visible style choice affect both the model request
# and the deterministic finishing pass.  They are deliberately conservative:
# protection and source luminance remain authoritative over stylistic grading.
STYLE_RENDER_PROFILES = {
    "original_ink": {
        "style_guidance": (
            "clean original ink colouring, restrained chroma, minimal rendering change"
        ),
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "saturation": 0.78,
        "contrast": 0.98,
        "hue_shift": 0.0,
        "chroma_multiplier": 0.84,
    },
    "modern_anime": {
        "style_guidance": (
            "modern cel-shaded anime rendering, crisp separated flat colours and controlled shadows"
        ),
        "guidance_scale": 1.05,
        "num_inference_steps": 5,
        "saturation": 1.18,
        "contrast": 1.06,
        "hue_shift": 0.0,
        "chroma_multiplier": 1.08,
    },
    "soft_painted": {
        "style_guidance": (
            "soft painted illustration, smooth material transitions and gentle brush texture"
        ),
        "guidance_scale": 1.0,
        "num_inference_steps": 5,
        "saturation": 0.94,
        "contrast": 0.93,
        "hue_shift": 0.0,
        "chroma_multiplier": 1.0,
    },
    "watercolor": {
        "style_guidance": (
            "transparent watercolor wash, paper-like pigment variation and airy low-contrast colour"
        ),
        "guidance_scale": 0.98,
        "num_inference_steps": 5,
        "saturation": 0.80,
        "contrast": 0.88,
        "hue_shift": 0.0,
        "chroma_multiplier": 0.84,
    },
    "dramatic_noir": {
        "style_guidance": (
            "dramatic noir cinema, high-contrast graphic lighting with sparse accent colours"
        ),
        "guidance_scale": 1.08,
        "num_inference_steps": 5,
        "saturation": 0.72,
        "contrast": 1.16,
        "hue_shift": 0.0,
        "chroma_multiplier": 0.76,
    },
}

COLOR_RENDER_PROFILES = {
    "natural": {"saturation": 1.0, "contrast": 1.0, "hue_shift": 0.0},
    "warm_cinematic": {"saturation": 1.02, "contrast": 1.02, "hue_shift": 5.0},
    "cool_night": {"saturation": 0.92, "contrast": 1.04, "hue_shift": -10.0},
    "vivid_anime": {"saturation": 1.14, "contrast": 1.04, "hue_shift": 0.0},
    "pastel": {"saturation": 0.78, "contrast": 0.90, "hue_shift": 2.0},
    "retro_print": {"saturation": 0.86, "contrast": 0.96, "hue_shift": 7.0},
}


def get_color_preset(preset_id: str) -> PromptPreset:
    try:
        return COLOR_PRESETS[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown color preset: {preset_id}") from exc


def get_style_preset(preset_id: str) -> PromptPreset:
    try:
        return STYLE_PRESETS[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown style preset: {preset_id}") from exc


def render_profile(color_preset_id: str, style_preset_id: str) -> dict[str, float | int | str]:
    if color_preset_id not in COLOR_RENDER_PROFILES:
        raise ValueError(f"Unknown color preset: {color_preset_id}")
    if style_preset_id not in STYLE_RENDER_PROFILES:
        raise ValueError(f"Unknown style preset: {style_preset_id}")
    style = STYLE_RENDER_PROFILES[style_preset_id]
    color = COLOR_RENDER_PROFILES[color_preset_id]
    return {
        **style,
        "saturation": float(style["saturation"]) * float(color["saturation"]),
        "contrast": float(style["contrast"]) * float(color["contrast"]),
        "hue_shift": float(style["hue_shift"]) + float(color["hue_shift"]),
    }


def build_prompt(
    mode: JobMode,
    color_preset_id: str,
    style_preset_id: str,
    user_prompt: str,
) -> str:
    color = get_color_preset(color_preset_id)
    style = get_style_preset(style_preset_id)
    operation = (
        "Colorize this exact black-and-white manga image"
        if mode == JobMode.COLORIZE
        else "Restyle this exact manga image"
    )
    constraints = (
        "Preserve every character identity, face, expression, body pose, hand, object, "
        "camera angle, panel boundary, speech bubble, text location and crop. "
        "Do not add, remove, rewrite or relocate content. "
        "Keep skin, face, eyes, mouth, hands, feet, hair and clothing as separate "
        "semantic regions. "
        "Never paint exposed skin with clothing color or merge a garment across a body boundary. "
        "Reuse the same character and object colors across panels and pages. "
        "Keep speech bubbles, lettering, sound effects, ink lines and frame borders unchanged."
    )
    parts = [operation, constraints, color.prompt, style.prompt]
    if user_prompt.strip():
        parts.append(user_prompt.strip())
    return ". ".join(parts)


def presets_payload() -> dict[str, list[dict[str, str]]]:
    return {
        "colors": [preset.to_json_dict() for preset in COLOR_PRESETS.values()],
        "styles": [preset.to_json_dict() for preset in STYLE_PRESETS.values()],
    }
