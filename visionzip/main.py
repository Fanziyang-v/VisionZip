"""
Copy from https://github.com/dvlab-research/VisionZip/blob/main/visionzip/main.py

Modified by Fanziyang-v
"""
from .utils import clip_encoder_layer_forward, clip_attn_forward, apply_info
from .clip_encoder import clip_vision_tower_feature_select, clip_vision_tower_forward
from .llava_arch import prepare_inputs_labels_for_multimodal_visionzip, encode_images_visionzip, encode_images_visionzip_multi, restore_image_features_sorted

def visionzip(model, dominant=108, contextual=20):

    apply_info(model.model.vision_tower.vision_tower, dominant_num=dominant, contextual_num=contextual)

    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention
    CLIPEncoderLayer.forward = clip_encoder_layer_forward
    CLIPAttention.forward = clip_attn_forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.feature_select = clip_vision_tower_feature_select
    CLIPVisionTower.forward = clip_vision_tower_forward

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip

    return model
