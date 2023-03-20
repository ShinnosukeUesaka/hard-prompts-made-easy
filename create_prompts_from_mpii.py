import open_clip
from optim_utils import * 

import torch
import argparse

from pathlib import Path
from PIL import Image
import os
import h5py



def get_normalized_image_mpii(dataset_dir: Path = Path("/work/jqin/diffusion_iccv/xgaze_512/train"), subject_number: int = 0, image_index = 0, random_sample=False, max_image_index=100):
    max_try = 20
    tries = 0
    
    if random_sample:
        while True: 
            image_index = random.randint(0,max_image_index)
            subject_path = random.choice(list(dataset_dir.iterdir()))
            
            if subject_path.suffix == ".h5":
                break
            if tries > max_try:
                raise Exception
            tries += 1
    else:
        subject_path = dataset_dir / f"p{subject_number:02}.h5"
    assert subject_path.is_file()
    print(subject_path)
    with h5py.File(subject_path, 'r', libver='latest', swmr=True) as f:
        image = f['face_patch'][image_index]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = np.uint8(image)
        print(f"Image Index: {image_index}, Path: {subject_path}")

        
    return Image.fromarray(image)


def main(num_images_per_generation: int, num_generations: int, output_file: Path, mpii_dir: Path):
    args = argparse.Namespace()
    args.__dict__.update(read_json("sample_config.json"))
    args

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)


    for i in range(num_generations):
        MAX_IMAGE_INDEX = 2500
        
        subject_indices = random.sample(range(0, 14), num_images_per_generation)
        
        images = []
        for subject_index in subject_indices:
            image = get_normalized_image_mpii(dataset_dir=mpii_dir, subject_number=subject_index, random_sample=False)
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
        output_file=Path("hard_prompts.json"),
        mpii_dir=Path("/home/s-uesaka/normalization_scripts/out")
    )