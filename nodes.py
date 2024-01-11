import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter
import uuid
from scipy.interpolate import interp1d, PchipInterpolator
import torchvision
from utils import *
import folder_paths
import json

output_dir = "outputs"
ensure_dirname(output_dir)

def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    # fx = interp1d(t, x, kind='cubic')
    # fy = interp1d(t, y, kind='cubic')
    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points

def visualize_drag_v2(background_image_path, splited_tracks, width, height):
    trajectory_maps = []
    
    background_image = Image.open(background_image_path).convert('RGBA')
    background_image = background_image.resize((width, height))
    w, h = background_image.size
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128
    transparent_background = Image.fromarray(transparent_background)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
            for i in range(len(splited_track)-1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i+1][0]), int(splited_track[i+1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2)
                if i == len(splited_track)-2:
                    cv2.arrowedLine(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2)
        else:
            cv2.circle(transparent_layer, (int(splited_track[0][0]), int(splited_track[0][1])), 5, (255, 0, 0, 192), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
    trajectory_maps.append(trajectory_map)
    return trajectory_maps, transparent_layer

class Drag:
    def __init__(self, device, model_path, cfg_path, height, width, model_length):
        self.device = device
        cf = import_filename(cfg_path)
        Net, args = cf.Net, cf.args
        drag_nuwa_net = Net(args)
        state_dict = file2data(model_path, map_location='cpu')
        adaptively_load_state_dict(drag_nuwa_net, state_dict)
        drag_nuwa_net.eval()
        drag_nuwa_net.to(device)
        # drag_nuwa_net.half()
        self.drag_nuwa_net = drag_nuwa_net
        self.height = height
        self.width = width
        _, model_step, _ = split_filename(model_path)
        self.ouput_prefix = f'{model_step}_{width}X{height}'
        self.model_length = model_length

    @torch.no_grad()
    def forward_sample(self, input_drag, input_first_frame, motion_bucket_id, outputs=dict()):
        device = self.device
    
        b, l, h, w, c = input_drag.size()
        drag = self.drag_nuwa_net.apply_gaussian_filter_on_drag(input_drag)
        drag = torch.cat([torch.zeros_like(drag[:, 0]).unsqueeze(1), drag], dim=1)  # pad the first frame with zero flow
        drag = rearrange(drag, 'b l h w c -> b l c h w')

        input_conditioner = dict()
        input_conditioner['cond_frames_without_noise'] = input_first_frame
        input_conditioner['cond_frames'] = (input_first_frame + 0.02 * torch.randn_like(input_first_frame))
        input_conditioner['motion_bucket_id'] = torch.tensor([motion_bucket_id]).to(drag.device).repeat(b * (l+1))
        input_conditioner['fps_id'] = torch.tensor([self.drag_nuwa_net.args.fps]).to(drag.device).repeat(b * (l+1))
        input_conditioner['cond_aug'] = torch.tensor([0.02]).to(drag.device).repeat(b * (l+1))

        input_conditioner_uc = {}
        for key in input_conditioner.keys():
            if key not in input_conditioner_uc and isinstance(input_conditioner[key], torch.Tensor):
                input_conditioner_uc[key] = input_conditioner[key].clone()
        
        c, uc = self.drag_nuwa_net.conditioner.get_unconditional_conditioning(
            input_conditioner,
            batch_uc=input_conditioner_uc,
            force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
        )

        for k in ["crossattn", "concat"]:
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=self.drag_nuwa_net.num_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...")
            c[k] = repeat(c[k], "b ... -> b t ...", t=self.drag_nuwa_net.num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...")
    
        H, W = input_conditioner['cond_frames_without_noise'].shape[2:]
        shape = (self.drag_nuwa_net.num_frames, 4, H // 8, W // 8)
        randn = torch.randn(shape).to(self.device)

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            2, self.drag_nuwa_net.num_frames
        ).to(self.device)
        additional_model_inputs["num_video_frames"] = self.drag_nuwa_net.num_frames
        additional_model_inputs["flow"] = drag.repeat(2, 1, 1, 1, 1)    # c and uc

        def denoiser(input, sigma, c):
            return self.drag_nuwa_net.denoiser(self.drag_nuwa_net.model, input, sigma, c, **additional_model_inputs)
        
        samples_z = self.drag_nuwa_net.sampler(denoiser, randn, cond=c, uc=uc)
        samples = self.drag_nuwa_net.decode_first_stage(samples_z)

        outputs['logits_imgs'] = rearrange(samples, '(b l) c h w -> b l c h w', b=b)
        return outputs

    def run(self, first_frame, tracking_points, inference_batch_size, motion_bucket_id):
        original_width, original_height=576, 320

        input_all_points = tracking_points
        resized_all_points = [tuple([tuple([int(e1[0]*self.width/original_width), int(e1[1]*self.height/original_height)]) for e1 in e]) for e in input_all_points]

        input_drag = torch.zeros(self.model_length - 1, self.height, self.width, 2)
        for splited_track in resized_all_points:
            if len(splited_track) == 1: # stationary point
                displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            splited_track = interpolate_trajectory(splited_track, self.model_length)
            splited_track = splited_track[:self.model_length]
            if len(splited_track) < self.model_length:
                splited_track = splited_track + [splited_track[-1]] * (self.model_length -len(splited_track))
            for i in range(self.model_length - 1):
                start_point = splited_track[i]
                end_point = splited_track[i+1]
                input_drag[i][int(start_point[1])][int(start_point[0])][0] = end_point[0] - start_point[0]
                input_drag[i][int(start_point[1])][int(start_point[0])][1] = end_point[1] - start_point[1]

        image_pil = first_frame.resize((self.width, self.height), Image.BILINEAR).convert('RGB')
        
        visualized_drag, _ = visualize_drag_v2(first_frame_path, resized_all_points, self.width, self.height)
        
        first_frames_transform = transforms.Compose([
                        lambda x: Image.fromarray(x),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
        
        outputs = None
        ouput_video_list = []
        num_inference = 1
        for i in tqdm(range(num_inference)):
            if not outputs:
                first_frames = pil2arr(first_frame)
                first_frames = repeat(first_frames_transform(first_frames), 'c h w -> b c h w', b=inference_batch_size).to(self.device)
            else:
                first_frames = outputs['logits_imgs'][:, -1]
            
            outputs = self.forward_sample(
                                            repeat(input_drag[i*(self.model_length - 1):(i+1)*(self.model_length - 1)], 'l h w c -> b l h w c', b=inference_batch_size).to(self.device), 
                                            first_frames,
                                            motion_bucket_id)
            ouput_video_list.append(outputs['logits_imgs'])

        for i in range(inference_batch_size):
            ouput_tensor = [ouput_video_list[0][i]]
            for j in range(num_inference - 1):
                ouput_tensor.append(ouput_video_list[j+1][i][1:])
            ouput_tensor = torch.cat(ouput_tensor, dim=0)
            outputs_path = os.path.join(output_dir, f'output_{i}_{id}.gif')
            data2file([transforms.ToPILImage('RGB')(utils.make_grid(e.to(torch.float32).cpu(), normalize=True, range=(-1, 1))) for e in ouput_tensor], outputs_path,
                      printable=False, duration=1 / 6, override=True)

        return visualized_drag[0], outputs_path

class LoadCheckPointDragNUWA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "drag_nuwa_svd.pth"}),
                "height": ("INT", {"default": 320}),
                "width": ("INT", {"default": 576}),
                "model_length": ("INT", {"default": 14}),
            }
        }
        
    RETURN_TYPES = ("DragNUWA",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_dragnuwa"
    CATEGORY = "DragNUWA"
    
    def load_dragnuwa(self, ckpt_name, height, width, model_length):
        comfy_path = os.path.dirname(folder_paths.__file__)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        DragNUWA_net = Drag("cuda:0", ckpt_path, 'DragNUWA_net.py', height, width, model_length)
        return (DragNUWA_net,)

class DragNUWARun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tracking_points": ("STRING", {"multiline": True, "default":"[[[25,25],[128,128]]]"}),
                "inference_batch_size": ("INT", {"default": 1, "min": 1, "max": 1}),
                "motion_bucket_id": ("INT", {"default": 4, "min": 1, "max": 100}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "DragNUWA"
    
    def run_inference(self, image, tracking_points):
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        tracking_points=json.loads(tracking_points)
        DragNUWA_net.run(image, tracking_points, inference_batch_size, motion_bucket_id)
    
