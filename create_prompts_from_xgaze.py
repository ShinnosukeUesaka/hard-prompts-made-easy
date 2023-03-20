import open_clip
from optim_utils import * 

import torch
import argparse

from pathlib import Path
from PIL import Image
import os
import h5py



def get_normalized_image(subject_path: Path = Path("/work/jqin/diffusion_iccv/xgaze_512/train/0000.h5"), max_image_index=100):
    
    image_index = random.randint(0,max_image_index)
 
    assert subject_path.is_file()
    with h5py.File(subject_path, 'r', libver='latest', swmr=True) as f:
        image = f['face_patch'][image_index]
        print(f"Image Index: {image_index}, Path: {subject_path} CameraIndex: {f['cam_index'][image_index]}")

        
    return Image.fromarray(image[:,:,::-1])


def main(num_images_per_generation: int, num_generations: int, output_file: Path, dataset_dir: Path):
    args = argparse.Namespace()
    args.__dict__.update(read_json("sample_config.json"))
    args

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)


    for i in range(num_generations):
        MAX_IMAGE_INDEX = 2500
        
        temp_subject_paths = list(dataset_dir.iterdir())
        all_subject_paths = []
        for subject_path in temp_subject_paths:
            if subject_path.suffix == ".h5":
                all_subject_paths.append(subject_path)
        subject_paths = random.sample(all_subject_paths, num_images_per_generation)
  
        images = []
        for subject_path in subject_paths:
            image = get_normalized_image(subject_path=subject_path)
            images.append(image)

        learned_prompt, best_sim = optimize_prompt(model, preprocess, args, device, target_images=images)
        prompt = (
            {
                'prompt': learned_prompt,
                'sim': best_sim,
            }
        )
        
        if output_file.exists():
            with open(output_file, 'r') as f:
                prompts = json.load(f)
            prompts.append(
                prompt
            )
            with open(output_file, 'w') as f:
                json.dump(prompts, f)
        else:
            with open(output_file, 'w') as f:
                json.dump([prompt], f)
            
if __name__ == "__main__":
    main(
        num_generations=1000,
        num_images_per_generation=4,
        output_file=Path("hard_prompts_xgaze.json"),
        dataset_dir=Path("/home/s-uesaka/datasets/xgaze_512/train")
    )