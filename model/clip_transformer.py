import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config
        
        if self.config.huggingface:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            from model.clip_model import load_clip
            self.clip = load_clip(config.clip_arch)

        config.pooling_type = 'transformer'
        self.pool_frames = Transformer(config)


    def encode(self, data):
        """
        Per-sample CLIP features, without any interaction between samples.

        Output
            text_features: num_texts x embed_dim
            video_features: num_vids x num_frames x embed_dim
        """
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)

        if self.config.huggingface:
            text_features = self.clip.get_text_features(**text_data)
            video_features = self.clip.get_image_features(video_data)
        else:
            text_features = self.clip.encode_text(text_data)
            video_features = self.clip.encode_image(video_data)

        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        return text_features, video_features


    def pool(self, text_features, video_features):
        """
        Text-conditioned pooling over every text-video combination.

        Output
            video_features_pooled: num_vids x num_texts x embed_dim
        """
        return self.pool_frames(text_features, video_features)


    def forward(self, data, return_all_frames=False):
        # Same operations in the same order as before the encode/pool split.
        text_features, video_features = self.encode(data)
        video_features_pooled = self.pool(text_features, video_features)

        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled
