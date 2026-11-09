"""Create local-only sampled-frame contact sheets for the saved failure cases.

Frames are read from the user's existing dataset. Output defaults to .cache/ and
is not committed. Public analysis retains video IDs, ranks and attention plots.
"""

import argparse
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont
import torch

from datasets.video_capture import VideoCapture
from datasets.model_transforms import init_transform_dict


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/checkpoint_analysis_20261109')
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.cache/checkpoint_analysis_frames')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads((args.results_dir / 'casebook.json').read_text(encoding='utf-8'))
    font = ImageFont.truetype('arial.ttf', 17)
    small = ImageFont.truetype('arial.ttf', 14)
    positions = [0, 2, 4, 6, 8, 11]
    transform = init_transform_dict(224)['clip_test']
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    cache = {}
    metadata = []
    for case in cases:
        roles = {case['video_id']: ['ground truth']}
        captions = {case['video_id']: case['caption']}
        for label, row in case['models'].items():
            roles.setdefault(row['top1_video_id'], []).append(label + ' top-1')
            captions[row['top1_video_id']] = row['top1_caption']
        image = Image.new('RGB', (1240, 110 + len(roles)*220), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((14, 10), f'Case {case["case_id"]}: {case["selection"]} / query {case["query_index"]}', font=font, fill='black')
        draw.multiline_text((14, 36), textwrap.fill(case['caption'], 112), font=font, fill='black', spacing=4)
        draw.text((14, 86), 'Local only: actual 224x224 evaluation crops, normalization inverted; six of twelve uniform positions.', font=small, fill='#555555')
        for row_index, (video, video_roles) in enumerate(roles.items()):
            if video not in cache:
                frames, indices = VideoCapture.load_frames_from_video(str(args.source_root / 'data/MSRVTT/vids' / (video + '.mp4')), 12, 'uniform')
                frames = (transform(frames) * clip_std + clip_mean).clamp(0, 1)
                cache[video] = (frames, [int(i) for i in indices])
            frames, indices = cache[video]
            y = 110 + row_index*220
            draw.text((14, y), video + ': ' + ', '.join(video_roles), font=font, fill='#153759')
            draw.multiline_text((14, y+23), textwrap.fill('Paired evaluation caption: ' + captions[video], 135), font=small, fill='black', spacing=2)
            for col, position in enumerate(positions):
                array = (frames[position].permute(1, 2, 0).numpy()*255).round().astype('uint8')
                thumb = Image.fromarray(array)
                thumb.thumbnail((196, 132), Image.LANCZOS)
                x = 14 + col*204
                image.paste(thumb, (x + (196-thumb.width)//2, y+67))
                draw.text((x, y+201), f'pos {position+1}, frame {indices[position]}', font=small, fill='black')
            metadata.append({'case_id': case['case_id'], 'video_id': video, 'roles': video_roles, 'all_sampled_frame_indices': indices, 'displayed_positions_1based': [x+1 for x in positions],
                             'display_transform': 'Original clip_test resize + 224x224 center crop; CLIP normalization inverted for display.'})
        image.save(args.output_dir / f'case_{case["case_id"]:02d}.png')
    (args.results_dir / 'case_frame_indices.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(f'Created {len(cases)} local contact sheets; exported frame indices without image/video media.')


if __name__ == '__main__':
    main()
