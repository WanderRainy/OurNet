import torch


def load_weights(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = state.get('model', state.get('state_dict', state))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
