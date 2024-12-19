import torch
import random

def rank_video(video_tensor, model):
    # return random.random(), 31
    model.eval()
    with torch.no_grad():
        output = model(video_tensor)
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(output)
    top_probs, top_classes = torch.topk(probs, 2)
    return top_probs[0][0].item(), top_classes[0][0].item()