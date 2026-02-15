import torch
import numpy as np
from model import ActionSpottingTransformerNet

def nms(events, t=30):
    events = sorted(events, key=lambda x: -x[1])
    keep = []
    for e in events:
        if all(abs(e[0]-k[0]) > t for k in keep):
            keep.append(e)
    return keep

def infer(video_feat):
    model = ActionSpottingTransformerNet().cuda().eval()
    feats = np.load(video_feat)
    preds = []

    for i in range(0, len(feats)-40, 20):
        x = torch.tensor(feats[i:i+40]).unsqueeze(0).cuda()
        c,_ = model(x)
        s = torch.softmax(c,1)
        cls = torch.argmax(s).item()
        conf = s[0,cls].item()
        if cls>0 and conf>0.5:
            preds.append((i/2, conf, cls))

    return nms(preds)
