# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Models for Text and Image Composition."""

import numpy as np
import torch
import torch.nn.functional as F
from .torch_functions import NormalizationLayer, TripletLoss
import torchvision
from clip_client import Client as BertClient
from torch.autograd import Variable
from transformers import ViTImageProcessor, ViTModel

from .text_model import TextLSTMModel

bc = BertClient("grpc://0.0.0.0:51000")


class ConCatModule(torch.nn.Module):
    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, 1)

        return x


class SelfAttentionModule(torch.nn.Module):
    def __init__(self, feature_size=512):
        super(SelfAttentionModule, self).__init__()
        self.fc_query = torch.nn.Linear(feature_size, feature_size)
        self.sqrt_dk = feature_size**0.5

    def forward(self, theta):
        query = self.fc_query(theta)
        attn_scores = torch.matmul(query, query.transpose(-1, -2)) / self.sqrt_dk
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, theta)

        output = torch.cat(
            (attn_output, theta), dim=-1
        )  # Concatenating the output with the original theta

        return output


class MultiheadCrossAttention(torch.nn.Module):
    def __init__(self, query_feature_size=512, key_value_feature_size=512, num_heads=8):
        super(MultiheadCrossAttention, self).__init__()
        assert (
            query_feature_size % num_heads == 0
        ), "Query feature size must be divisible by the number of heads."
        assert (
            key_value_feature_size % num_heads == 0
        ), "Key/Value feature size must be divisible by the number of heads."

        self.query_feature_size = query_feature_size
        self.key_value_feature_size = key_value_feature_size
        self.num_heads = num_heads
        self.head_dim = query_feature_size // num_heads

        # Linear transformations for queries, keys, and values for each head
        self.W_q = torch.nn.Linear(query_feature_size, query_feature_size)
        self.W_k = torch.nn.Linear(key_value_feature_size, key_value_feature_size)
        self.W_v = torch.nn.Linear(key_value_feature_size, key_value_feature_size)

        # Linear transformation for the concatenated outputs of all heads
        self.W_o = torch.nn.Linear(query_feature_size, query_feature_size)

    def split_heads(self, x):
        """
        Split the input tensor into multiple heads.
        Args:
            x (Tensor): Input tensor of shape (batch_size, feature_size).
        Returns:
            Tensor: Reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, _ = x.size()
        x = x.view(batch_size, self.num_heads, self.head_dim)
        
        # Add a singleton dimension at index 1
        x = x.unsqueeze(1)

        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, queries, keys, values):
        """
        Forward pass of the multi-head cross-attention module.
        Args:
            queries (Tensor): Query tensor of shape (batch_size, query_feature_size).
            keys (Tensor): Key tensor of shape (batch_size, key_value_feature_size).
            values (Tensor): Value tensor of shape (batch_size, key_value_feature_size).
        Returns:
            Tensor: Output tensor of shape (batch_size, query_seq_len, query_feature_size).
        """
        batch_size, _ = queries.size()

        # Linear transformations for queries, keys, and values
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # Split the queries, keys, and values into multiple heads
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        # Scaled dot-product attention
        attn_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)

        # Concatenate the outputs of all heads and apply a linear transformation
        attn_output = (
            attn_output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, -1, self.query_feature_size)
        )
        output = self.W_o(attn_output)

        return output


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, feature_size=512, num_heads=8):
        super(MultiheadSelfAttention, self).__init__()
        assert (
            feature_size % num_heads == 0
        ), "Feature size must be divisible by the number of heads."

        self.feature_size = feature_size
        self.num_heads = num_heads
        self.head_dim = feature_size // num_heads

        # Linear transformations for queries, keys, and values for each head
        self.W_q = torch.nn.Linear(feature_size, feature_size)
        self.W_k = torch.nn.Linear(feature_size, feature_size)
        self.W_v = torch.nn.Linear(feature_size, feature_size)

        # Linear transformation for the concatenated outputs of all heads
        self.W_o = torch.nn.Linear(feature_size, feature_size)

    def split_heads(self, x):
        """
        Split the input tensor into multiple heads.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, feature_size).
        Returns:
            Tensor: Reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        """
        Forward pass of the multi-head self-attention module.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, feature_size).
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, feature_size).
        """
        batch_size, seq_len, _ = x.size()

        # Linear transformations for queries, keys, and values
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # Split the queries, keys, and values into multiple heads
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        # Scaled dot-product attention
        attn_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)

        # Concatenate the outputs of all heads and apply a linear transformation
        attn_output = (
            attn_output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.feature_size)
        )
        output = self.W_o(attn_output)

        return output


class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super().__init__()
        self.normalization_layer = NormalizationLayer(
            normalize_scale=4.0, learn_scale=True
        )
        self.soft_triplet_loss = TripletLoss()

    #         self.name = 'model_name'

    def extract_img_feature(self, imgs):
        raise NotImplementedError

    def extract_text_feature(self, text_query, use_bert):
        raise NotImplementedError

    def compose_img_text(self, imgs, text_query):
        raise NotImplementedError

    def compute_loss(self, imgs_query, text_query, imgs_target, soft_triplet_loss=True):
        dct_with_representations = self.compose_img_text(imgs_query, text_query)
        composed_source_image = self.normalization_layer(dct_with_representations["repres"])
        target_img_features_non_norm = self.extract_img_feature(imgs_target)
        target_img_features = self.normalization_layer(target_img_features_non_norm)
        assert (
            composed_source_image.shape[0] == target_img_features.shape[0]
            and composed_source_image.shape[1] == target_img_features.shape[1]
        )
        # Get Rot_Sym_Loss
        if self.name == "composeAE":
            CONJUGATE = Variable(torch.cuda.FloatTensor(32, 1).fill_(-1.0), requires_grad=False)
            conjugate_representations = self.compose_img_text_features(
                target_img_features_non_norm,
                dct_with_representations["text_features"],
                CONJUGATE,
            )
            composed_target_image = self.normalization_layer(conjugate_representations["repres"])
            source_img_features = self.normalization_layer(
                dct_with_representations["img_features"]
            )  # img1
            if soft_triplet_loss:
                dct_with_representations["rot_sym_loss"] = self.compute_soft_triplet_loss_(
                    composed_target_image, source_img_features
                )
            else:
                dct_with_representations[
                    "rot_sym_loss"
                ] = self.compute_batch_based_classification_loss_(
                    composed_target_image, source_img_features
                )
        else:  # tirg, RealSpaceConcatAE etc
            dct_with_representations["rot_sym_loss"] = 0

        if soft_triplet_loss:
            return (
                self.compute_soft_triplet_loss_(composed_source_image, target_img_features),
                dct_with_representations,
            )
        else:
            return (
                self.compute_batch_based_classification_loss_(
                    composed_source_image, target_img_features
                ),
                dct_with_representations,
            )

    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert triplets and len(triplets) < 2000
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)


class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
    """Base class for image and text encoder."""

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__()
        # img model
        img_model = torchvision.models.resnet18(pretrained=True)
        self.name = name

        class GlobalAvgPool2d(torch.nn.Module):
            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d()
        img_model.fc = torch.nn.Sequential(torch.nn.Linear(image_embed_dim, image_embed_dim))
        self.img_model = img_model

        # text model
        self.text_model = TextLSTMModel(
            texts_to_build_vocab=text_query,
            word_embed_dim=text_embed_dim,
            lstm_hidden_dim=text_embed_dim,
        )

    def extract_img_feature(self, imgs):
        return self.img_model(imgs)

    def extract_text_feature(self, text_query, use_bert):
        if use_bert:
            text_features = bc.encode(text_query)
            return torch.from_numpy(text_features).cuda()
        return self.text_model(text_query)


class TIRG(ImgEncoderTextEncoderBase):
    """The TIRG model.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        merged_dim = image_embed_dim + text_embed_dim

        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, image_embed_dim),
        )

        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, merged_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(merged_dim, image_embed_dim),
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        f1 = self.gated_feature_composer((img_features, text_features))
        f2 = self.res_info_composer((img_features, text_features))
        f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]

        dct_with_representations = {"repres": f}
        return dct_with_representations


class ComplexProjectionModule(torch.nn.Module):
    # def __init__(self, image_embed_dim =512, text_embed_dim = 768):
    def __init__(self, image_embed_dim=512, text_embed_dim=512):
        super().__init__()
        self.bert_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim),
            torch.nn.Linear(text_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        x1 = self.image_features(x[0])
        x2 = self.bert_features(x[1])
        # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
        # which results in the CONJUGATE of text features in the complex space
        CONJUGATE = x[2]
        num_samples = x[0].shape[0]
        CONJUGATE = CONJUGATE[:num_samples]
        delta = x2  # text as rotation
        re_delta = torch.cos(delta)
        im_delta = CONJUGATE * torch.sin(delta)

        re_score = x1 * re_delta
        im_score = x1 * im_delta

        concat_x = torch.cat([re_score, im_score], 1)
        x0copy = x[0].unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        re_score = re_score.unsqueeze(1)
        im_score = im_score.unsqueeze(1)

        return concat_x, x1, x2, x0copy, re_score, im_score


class LinearMapping(torch.nn.Module):
    """
    This is linear mapping to image space. rho(.)
    """

    def __init__(self, image_embed_dim=512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        theta_linear = self.mapping(x[0])
        return theta_linear


class ConvMapping(torch.nn.Module):
    """
    This is convoultional mapping to image space. rho_conv(.)
    """

    def __init__(self, image_embed_dim=512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        # in_channels, output channels
        self.conv = torch.nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(16)

    def forward(self, x):
        concat_features = torch.cat(x[1:], 1)
        concat_x = self.conv(concat_features)
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 1024))
        theta_conv = self.mapping(final_vec)
        return theta_conv


class ComposeAE(ImgEncoderTextEncoderBase):
    """The ComposeAE model.

    The method is described in
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        # merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(ComplexProjectionModule(), LinearMapping())
        self.encoderWithConv = torch.nn.Sequential(ComplexProjectionModule(), ConvMapping())
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim),
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(
        self,
        img_features,
        text_features,
        CONJUGATE=Variable(torch.cuda.FloatTensor(32, 1).fill_(1.0), requires_grad=False),
    ):
        theta_linear = self.encoderLinear((img_features, text_features, CONJUGATE))
        theta_conv = self.encoderWithConv((img_features, text_features, CONJUGATE))

        theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {
            "repres": theta,
            "repr_to_compare_with_source": self.decoder(theta),
            "repr_to_compare_with_mods": self.txtdecoder(theta),
            "img_features": img_features,
            "text_features": text_features,
        }

        return dct_with_representations


class CAET(ImgEncoderTextEncoderBase):
    """
        The Compose AutoEncoder Transformer model.
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(ComplexProjectionModule(), LinearMapping())
        self.encoderWithConv = torch.nn.Sequential(ComplexProjectionModule(), ConvMapping())
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim),
        )

        # Initialize the ViT feature extractor and model
        self.vit_feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit_model.to(self.device)
        # Initialize the dimensionality reduction layer
        self.dim_reduction_layer = torch.nn.Linear(768, 512).to(self.device)

        # Create an instance of MultiheadCrossAttention
        self.cross_attention = MultiheadCrossAttention(
            query_feature_size=image_embed_dim,
            key_value_feature_size=text_embed_dim,
            num_heads=8,
        )

    def extract_img_feature(self, imgs):
        # This method assumes imgs is a list of PIL Images or paths to the images
        # You may need to adjust the method to suit the format of your imgs input
        inputs = self.vit_feature_extractor(images=imgs, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.vit_model(**inputs)
        # Now outputs.pooler_output contains the feature vectors for the images, you can return it or further process it
        outputs = self.dim_reduction_layer(outputs.pooler_output)
        return outputs

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(
        self,
        img_features,
        text_features,
        CONJUGATE=Variable(torch.cuda.FloatTensor(32, 1).fill_(1.0), requires_grad=False),
    ):
        theta_linear = self.encoderLinear((img_features, text_features, CONJUGATE))
        theta_conv = self.encoderWithConv((img_features, text_features, CONJUGATE))

        # Apply cross-attention here
        # theta_cross_attention = self.cross_attention(theta_linear, theta_conv, theta_conv)

        theta = theta_linear * self.a[1] + theta_conv * self.a[0]
        # theta = theta_cross_attention * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {
            "repres": theta,
            "repr_to_compare_with_source": self.decoder(theta),
            "repr_to_compare_with_mods": self.txtdecoder(theta),
            "img_features": img_features,
            "text_features": text_features,
        }

        return dct_with_representations


class RealConCatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        concat_x = torch.cat(x, -1)
        return concat_x


class RealLinearMapping(torch.nn.Module):
    """
    This is linear mapping from real space to image space.
    """

    # def __init__(self, image_embed_dim=512, text_embed_dim=768):
    def __init__(self, image_embed_dim=512, text_embed_dim=512):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim + image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim + image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        theta_linear = self.mapping(x)
        return theta_linear


class RealConvMapping(torch.nn.Module):
    """
    This is convoultional mapping from Real space to image space.
    """

    def __init__(self, image_embed_dim=512, text_embed_dim=768):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim + image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim + image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        # in_channels, output channels
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(20)

    def forward(self, x):
        concat_x = self.conv1(x.unsqueeze(1))
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 1280))
        theta_conv = self.mapping(final_vec)
        return theta_conv


class RealSpaceConcatAE(ImgEncoderTextEncoderBase):
    """The RealSpaceConcatAE model.

    The method  in ablation study Table 5 (Concat in real space)
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, text_query, image_embed_dim, text_embed_dim, use_bert, name):
        super().__init__(text_query, image_embed_dim, text_embed_dim, use_bert, name)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.use_bert = use_bert

        # merged_dim = image_embed_dim + text_embed_dim

        self.encoderLinear = torch.nn.Sequential(RealConCatModule(), RealLinearMapping())
        self.encoderWithConv = torch.nn.Sequential(RealConCatModule(), RealConvMapping())
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, text_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(text_embed_dim, text_embed_dim),
        )

    def compose_img_text(self, imgs, text_query):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(text_query, self.use_bert)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        theta_linear = self.encoderLinear((img_features, text_features))
        theta_conv = self.encoderWithConv((img_features, text_features))
        theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {
            "repres": theta,
            "repr_to_compare_with_source": self.decoder(theta),
            "repr_to_compare_with_mods": self.txtdecoder(theta),
            "img_features": img_features,
            "text_features": text_features,
        }

        return dct_with_representations
