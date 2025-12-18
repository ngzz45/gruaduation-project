
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
import math

# import kornia
from transformers import set_seed

import random
from .pca import PCA
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import List, Tuple

def process_image(image_processor, image_raw):
    answer = image_processor(image_raw)

    # Check if the result is a dictionary and contains 'pixel_values' key
    if 'pixel_values' in answer:
        answer = answer['pixel_values'][0]
    
    # Convert numpy array to torch tensor if necessary
    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)
    
    # If it's already a tensor, return it directly
    elif isinstance(answer, torch.Tensor):
        return answer
    
    else:
        raise ValueError("Unexpected output format from image_processor.")
    
    return answer

def mask_patches(tensor, indices, patch_size=14):
    """
    Creates a new tensor where specified patches are set to the mean of the original tensor.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (C, H, W)
    indices (list of int): Indices of the patches to modify
    patch_size (int): Size of one side of the square patch
    
    Returns:
    torch.Tensor: New tensor with modified patches
    """
    # Clone the original tensor to avoid modifying it
    new_tensor = tensor.clone()

    # Calculate the mean across the spatial dimensions
    mean_values = tensor.mean(dim=(1, 2), keepdim=True)
    
    # Number of patches along the width
    patches_per_row = tensor.shape[2] // patch_size
    total_patches = (tensor.shape[1] // patch_size) * (tensor.shape[2] // patch_size)


    for index in indices:
        # Calculate row and column position of the patch
        row = index // patches_per_row
        col = index % patches_per_row

        # Calculate the starting pixel positions
        start_x = col * patch_size
        start_y = row * patch_size

        # Replace the patch with the mean values
        new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = mean_values.expand(-1, patch_size, patch_size)#new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size].mean(dim=(1, 2), keepdim=True).expand(-1, patch_size, patch_size)# mean_values.expand(-1, patch_size, patch_size)

    return new_tensor


def get_prompts(args, model, tokenizer, data_demos, question, model_type):
    if model_type == "llava-1.5":
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        qs_pos = question
        qs_neg = question

        if hasattr(model.config, 'mm_use_im_start_end'):

            if model.config.mm_use_im_start_end:
                qs_pos = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_pos
            else:
                qs_pos = DEFAULT_IMAGE_TOKEN + '\n' + qs_pos

            if model.config.mm_use_im_start_end:
                qs_neg = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_neg
            else:
                qs_neg = DEFAULT_IMAGE_TOKEN + '\n' + qs_neg

            conv_pos = conv_templates[args.conv_mode].copy()
            conv_pos.append_message(conv_pos.roles[0], qs_pos)
            conv_pos.append_message(conv_pos.roles[1], None)
            conv_neg = conv_templates[args.conv_mode].copy()
            conv_neg.append_message(conv_neg.roles[0], qs_neg)
            conv_neg.append_message(conv_neg.roles[1], None)


            prompts_positive  = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            prompts_negative  = [conv_neg.get_prompt() + k['h_value'] for k in data_demos]

            input_ids_positive = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_positive]
            input_ids_negative = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_negative]

        
        else:
            from transformers import InstructBlipProcessor
            processor = InstructBlipProcessor.from_pretrained("../download_models/vicuna-7b-v1.1")

            input_ids_positive = []
            input_ids_negative = []

            for k in data_demos:
                image_path = os.path.join(args.data_file, 'train2014', k['image'])

                image_raw = Image.open(image_path).convert("RGB")
                input_ids_positive.append(processor(images=image_raw, text=question + k['value'], return_tensors="pt").to(model.device))
                input_ids_negative.append(processor(images=image_raw, text=question + k['h_value'], return_tensors="pt").to(model.device))

        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    
    elif model_type == "instructblip":
        from minigpt4.common.registry import registry
        from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor, BlipCaptionProcessor
        # 必须先准备 lavis processor
        vis_proc = Blip2ImageEvalProcessor().from_config()
        txt_proc = BlipCaptionProcessor().from_config()
        device = model.device

        input_ids_positive = []
        input_ids_negative = []

        for k in data_demos:
            # 读取 COCO 训练集图像
            image_path = os.path.join(args.data_file, "train2014", k['image'])
            raw_image = Image.open(image_path).convert("RGB")

            # 预处理图像
            image_tensor = vis_proc(raw_image).unsqueeze(0).to(device)

            # 处理文本
            text_pos = txt_proc(question + k['value'])
            text_neg = txt_proc(question + k['h_value'])

            # 保存为 dict，供 VTI 使用
            input_ids_positive.append({"image": image_tensor, "text": text_pos})
            input_ids_negative.append({"image": image_tensor, "text": text_neg})

        # 返回 (neg, pos) 配对
        inputs = tuple((input_ids_negative[i], input_ids_positive[i]) for i in range(len(data_demos)))
    
    else:

        prompts_positive = []
        prompts_negative = []

        for k in data_demos:
            image_path = os.path.join(args.data_file, 'train2014', k['image'])    
            prompts_positive.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['value']}]))
            prompts_negative.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['h_value']}]))

        input_ids_positive = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_positive]
        input_ids_negative = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_negative]
        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    return inputs

def get_demos(args, image_processor, model, tokenizer, model_type, patch_size = 14, file_path = './experiments/data/hallucination_vti_demos.jsonl'): 
    # Initialize a list to store the JSON objects
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)

    inputs_images = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(image_processor, image_raw)
        image_tensor_cd_all_trials = []

        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                
            image_tensor_cd_all_trials.append(image_tensor_cd)

        inputs_images.append([image_tensor_cd_all_trials, image_tensor])

    input_ids = get_prompts(args, model, tokenizer, data_demos, question, model_type=model_type)
    
    return inputs_images, input_ids


def get_hiddenstates(model, inputs, image_tensor):
        h_all = []
        with torch.no_grad():
            for example_id in range(len(inputs)):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs[example_id])):
                    if image_tensor is None:
                        h = model(
                                **inputs[example_id][style_id],
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                    else:
                        h = model(
                                inputs[example_id][style_id],
                                images=image_tensor[example_id][-1].unsqueeze(0).half(),
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True).hidden_states

                    embedding_token = []
                    for layer in range(len(h)):
                        embedding_token.append(h[layer][:,-1].detach().cpu())
                    
                    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all

def get_hiddenstates_blip(model, inputs, image_tensor):
    h_all = []

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for ex_id in range(len(inputs)):
        embeddings_for_all_styles = []

        for style_id in range(len(inputs[ex_id])):

            # ---- 视觉编码（保持不变）----
            img = image_tensor[ex_id][-1].unsqueeze(0).to(device=device, dtype=dtype)
            with torch.no_grad():
                image_embeds = model.ln_vision(model.visual_encoder(img))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

            # ---- QFormer 只用于产生 inputs_embeds，不输出 hidden_states ----
            query_tokens = model.query_tokens.expand(1, -1, -1).to(device)
            with torch.no_grad():
                q_outputs = model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                q_last = q_outputs.last_hidden_state[:, :query_tokens.size(1), :]
                q_proj = model.llm_proj(q_last)  # [1, 32, 4096]

            # ---- LLaMA tokens ----
            llm_tokens = model.llm_tokenizer(
                inputs[ex_id][style_id]["text"],
                return_tensors="pt",
            ).to(device)

            # 组合输入 embedding：query + text embedding
            text_embeds = model.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([q_proj, text_embeds], dim=1)
            attention_mask = torch.cat([
                torch.ones(q_proj.size()[:2], device=device, dtype=torch.long),
                llm_tokens.attention_mask
            ], dim=1)

            # ---- LLaMA 得到正确的 hidden_states（32 层）----
            with torch.no_grad():
                outputs = model.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                llm_hidden = outputs.hidden_states  # length = 33 (embed + 32 layers)

            # ---- 取每层 last token ----
            final_layers = [layer[:, -1].to(device) for layer in llm_hidden]

            embedding_token = torch.cat(final_layers, dim=0)  # [33, 4096]
            embeddings_for_all_styles.append(embedding_token.cpu())

        h_all.append(tuple(embeddings_for_all_styles))

    return h_all

def obtain_textual_vti(model, inputs, image_tensor, model_type, rank=1):
    # 1. 获取隐状态 (耗时操作，复用结果)
    if model_type == "instructblip":
        hidden_states = get_hiddenstates_blip(model, inputs, image_tensor)
    else:
        hidden_states = get_hiddenstates(model, inputs, image_tensor)
    
    hidden_states_all = []
    pos_all = [] # 存储正样本状态

    for i in range(len(hidden_states)):
        # 差异向量: x_pos - x_neg (用于PCA)
        diff = hidden_states[i][1].view(-1) - hidden_states[i][0].view(-1)
        hidden_states_all.append(diff)
        # 记录正样本状态 (用于计算 target_mean)
        pos_all.append(hidden_states[i][1]) 

    # 2. 计算 PCA 主方向 (u)
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    
    # 原始方向向量 (Layers * Dim)
    raw_direction = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(0)
    
    # 恢复形状 [Layers, Dim]
    L, D = hidden_states[0][0].shape
    direction = raw_direction.view(L, D)
    reading_direction = fit_data.mean(0).view(L, D)

    # 3. 计算目标投影均值 (\bar{z})
    # Stack 正样本: [Batch, Layers, Dim]
    pos_tensor = torch.stack(pos_all).to(direction.device).float()
    
    # 归一化方向向量 (投影需要单位向量)
    u_hat = F.normalize(direction, dim=-1)
    
    # 计算投影: z = x_pos · u_hat -> [Batch, Layers]
    # 对 Batch 维度求平均 -> [Layers]
    target_mean = torch.sum(pos_tensor * u_hat, dim=-1).mean(dim=0)

    return direction, reading_direction, target_mean

def average_tuples(tuples: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    # Check that the input list is not empty
    if not tuples:
        raise ValueError("The input list of tuples is empty.")

    # Check that all tuples have the same length
    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")

    # Initialize a list to store the averaged tensors
    averaged_tensors = []

    # Iterate over the indices of the tuples
    for i in range(n):
        # Stack the tensors at the current index and compute the average
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)

    # Convert the list of averaged tensors to a tuple
    averaged_tuple = tuple(averaged_tensors)

    return averaged_tuple

def get_visual_hiddenstates(model, image_tensor, model_type):
    h_all = []

    with torch.no_grad():
        if model_type == "llava-1.5":
            try:
                vision_model = model.model.vision_tower.vision_tower.vision_model
            except:
                vision_model = model.vision_model

        elif model_type == "instructblip":
            vision_model = model.visual_encoder
            vision_model.float()  

            # make it output all intermediate blocks
            # example: 0=input, 1~4 = blocks, 5=final norm
            vision_model.out_indices = tuple(range(len(vision_model.blocks) + 1))

        else:
            vision_model = model.transformer.visual
            model.transformer.visual.output_hidden_states = True
            

        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []

            for style_id in range(len(image_tensor[example_id])):

                # --- Case: list of augmented images ---
                if isinstance(image_tensor[example_id][style_id], list):
                    h_list = []

                    for image_tensor_ in image_tensor[example_id][style_id]:

                        if model_type == "llava-1.5":
                            h_ = vision_model(
                                image_tensor_.unsqueeze(0).half().cuda(),
                                output_hidden_states=True,
                                return_dict=True
                            ).hidden_states

                        elif model_type == "instructblip":
                            # returns a list of stage features
                            h_ = vision_model.forward_features(
                                image_tensor_.unsqueeze(0).cuda()
                            )

                        else:
                            _, h_ = vision_model(
                                image_tensor_.unsqueeze(0).cuda()
                            )

                        h_list.append(h_)

                    # average multi-views
                    h = average_tuples(h_list)

                # --- Case: single image ---
                else:
                    if model_type == "llava-1.5":
                        h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda(),
                            output_hidden_states=True,
                            return_dict=True
                        ).hidden_states

                    elif model_type == "instructblip":
                        h = vision_model.forward_features(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda()
                        )

                    else:
                        _, h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda()
                        )

                # unify all layer outputs into final tensor
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0)
                if embedding_token.dim() == 2:
                    embedding_token = embedding_token.unsqueeze(0)   # → (1, tokens, dim)
                embeddings_for_all_styles.append(embedding_token)

            h_all.append(tuple(embeddings_for_all_styles))

        if not (model_type == "llava-1.5" or model_type == "instructblip"):
            model.transformer.visual.output_hidden_states = False

    return h_all

def obtain_visual_vti(model, image_tensor, model_type,rank=1):

    hidden_states = get_visual_hiddenstates(model, image_tensor, model_type = model_type)
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)

    
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all,dim=1)[:] # n_token (no CLS token) x n_demos x D
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction
