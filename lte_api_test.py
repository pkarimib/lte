from lte_wrapper import SuperResolutionModel
import imageio
import numpy as np
import time
from argparse import ArgumentParser
import torch
import piq
from skimage import img_as_float32
from first_order_model.modules.model import Vgg19
from first_order_model.reconstruction import *
from first_order_model.utils import get_main_config_params

parser = ArgumentParser()
parser.add_argument("--config",
                    default="config/paper_configs/resolution512_with_hr_skip_connections.yaml",
                    help="path to config")
parser.add_argument("--checkpoint",
                    default='None',
                    help="path to the checkpoints")
parser.add_argument("--video-path",
                    default="512_kayleigh_10_second_0_1.mp4",
                    help="path to the video")
parser.add_argument("--log-dir",
                    default="./lte_api_test",
                    help="directory to save the results")
parser.add_argument("--output-name",
                    default="prediction",
                    help="name of the output file to be saved")
parser.add_argument("--output-fps",
                    default=30,
                    help="fps of the final video")
parser.add_argument("--lr-quantizer",
                    type=int, default=32,
                    help="quantizer to compress low-res video stream with")
parser.add_argument("--encode-lr",
                    action='store_true',
                    help="encode low-res video stream with vpx")
parser.set_defaults(verbose=False)
args = parser.parse_args()

main_configs = get_main_config_params(args.config)
generator_type = main_configs['generator_type']
use_lr_video = main_configs['use_lr_video']
lr_size = main_configs['lr_size']
print(main_configs)

video_duration = get_video_duration(args.video_path)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# model initialization and warm-up
model = SuperResolutionModel(args.config, args.checkpoint)
source_lr = np.random.rand(lr_size, lr_size, model.get_shape()[2])

for _ in range(1):
    _ = model.predict_with_lr_video(source_lr)

#get_model_info(args.log_dir, None,  model.generator)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timing_enabled = False#False if generator_type in ['vpx', 'bicubic'] else True
predictions = []
visual_metrics = []
loss_list = []
lr_stream = []
metrics_file = open(os.path.join(args.log_dir, args.output_name + '_metrics_summary.txt'), 'wt')
frame_metrics_file = open(os.path.join(args.log_dir, args.output_name + '_per_frame_metrics.txt'), 'wt')
write_in_file(frame_metrics_file, 'frame,psnr,ssim,ssim_db,lpips\n')
vgg_model = Vgg19()

if torch.cuda.is_available():
    vgg_model = vgg_model.cuda()
loss_fn_vgg = vgg_model.compute_loss

hr_encoder, lr_encoder = Vp8Encoder(), Vp8Encoder()
hr_decoder, lr_decoder = Vp8Decoder(), Vp8Decoder()
container = av.open(file=args.video_path, format=None, mode='r')
stream = container.streams.video[0]

if timing_enabled:
    generator_times, update_source_times = [], []

frame_idx = 0
for av_frame in container.decode(stream):
    # ground-truth
    frame = av_frame.to_rgb().to_ndarray()
    driving = frame
    driving_tensor = frame_to_tensor(img_as_float32(frame), device)
    driving_lr = resize_tensor_to_array(driving_tensor, lr_size, device)

    driving_lr_av = av.VideoFrame.from_ndarray(driving_lr)
    driving_lr_av.pts = av_frame.pts
    driving_lr_av.time_base = av_frame.time_base

    if args.encode_lr:
        driving_lr, compressed_tgt = get_frame_from_video_codec(driving_lr_av, lr_encoder,
                                                            lr_decoder, args.lr_quantizer)

    prediction = model.predict_with_lr_video(driving_lr)
    if args.encode_lr:
        lr_stream.append(compressed_tgt)

    if timing_enabled:
        generator_times.append(start.elapsed_time(end))

    prediction_tensor = frame_to_tensor(img_as_float32(prediction), device)
    loss_list.append(torch.abs(prediction_tensor - driving_tensor).mean().cpu().numpy())
    visual_metrics.append(Logger.get_visual_metrics(prediction_tensor, driving_tensor, loss_fn_vgg))
    predictions.append(prediction)
    if frame_idx % 100 == 0:
        print('total frames', frame_idx)
    frame_idx += 1
    if frame_idx > 30:
        break

imageio.mimsave(os.path.join(args.log_dir,
                args.output_name + '.mp4'),
                predictions, fps = int(args.output_fps))

lr_br = get_bitrate(lr_stream, video_duration)

for i, m in enumerate(visual_metrics):
    write_in_file(frame_metrics_file, f'{i},{m["psnr"][0]},{m["ssim"]},' +
                f'{m["ssim_db"]},{m["lpips"]}\n')
frame_metrics_file.close()

psnr, ssim, lpips_val, ssim_db = get_avg_visual_metrics(visual_metrics)
metrics_report = f'quantization:'

if args.encode_lr:
    metrics_report += f', lr_quantizer: {args.lr_quantizer}'

metrics_report += f'\nPSNR: {psnr}, SSIM: {ssim}, SSIM_DB: {ssim_db}, LPIPS: {lpips_val}, ' +\
    f'LR: {lr_br:.3f}Kbps \n' +\
    f'Reconstruction loss: {np.mean(loss_list)}\n'

if timing_enabled:
    metrics_report += f'generator: {np.average(generator_times)}'

write_in_file(metrics_file, metrics_report)
metrics_file.close()
