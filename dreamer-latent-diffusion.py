import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from datetime import datetime

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from cloudevents.sdk.event import v1
from dapr.clients import DaprClient
from dapr.ext.grpc import App, InvokeMethodRequest, InvokeMethodResponse
from dapr.clients.grpc._response import TopicEventResponse
import json
from types import SimpleNamespace
from minio import Minio

app = App()
globalModel = {}

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
globalModel = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")  # TODO: check path


@app.subscribe(pubsub_name='jetstream-pubsub', topic='request.latent-diffusion')
def generate(event: v1.Event) -> None:
    data = json.loads(event.Data())
    with DaprClient() as d:
        id = data["id"]
        req = json.loads(d.get_state(store_name="cosmosdb", key=id).data)
        if req == null or req["complete_time"]:
          return
        req["start_time"] = datetime.utcnow()
        d.save_state(store_name="cosmosdb", key=id, value=json.dumps(req))
        opt = SimpleNamespace(**data["input"])
        opt.outdir = os.path.join("/var/www/html/result/", id)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = globalModel.to(device)

        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        prompt = opt.prompt


        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        client = Minio("dumb.dev", access_key=os.getenv('NIGHTMAREBOT_MINIO_KEY'), secret_key=os.getenv('NIGHTMAREBOT_MINIO_SECRET'))
        sample_filenames=list()
        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.height//8, opt.width//8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        basename = f"{base_count:04}.png"
                        sample_filenames.append(basename)
                        filename = os.path.join(sample_path, basename)
                        Image.fromarray(x_sample.astype(np.uint8)).save(filename)
                        client.fput_object("nightmarebot-output", f"{id}/samples/{base_count:04}.png",filename, content_type="image/png")
                        base_count += 1
                    all_samples.append(x_samples_ddim)


        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        gridFilename = os.path.join(outpath, f'results.png')
        Image.fromarray(grid.astype(np.uint8)).save(gridFilename)
        client.fput_object("nightmarebot-output", f"{id}/results.png", gridFilename, content_type="image/png")
        d.publish_event(
            pubsub_name="jetstream-pubsub", 
            topic_name="response.latent-diffusion", 
            data=json.dumps({
                "id": data["id"], 
                "context": data["context"],
                "images": sample_filenames}),
            data_content_type="application/json")

        req.complete_time = datetime.utcnow()
        d.save_state(store_name="cosmosdb", key=id, value=json.dumps(req)) 
#        return InvokeMethodResponse(json.dumps(sample_filenames))
                    
app.run(50052)
