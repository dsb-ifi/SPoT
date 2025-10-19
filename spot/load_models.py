import torch
import timm
from quix import QuixDataset
from . import nn
from . import patch_dropout
from torchvision.transforms import v2
from torch.nn import Identity


valid_models = [
    'spot_mae_b16',
    'spot_21k_b16',
    'spot_1k_b16',
    'vit_mae_b16',
    'vit_21k_b16',
    'vit_1k_b16'
]

def load_trained_model(
    model_name: str, path:str|None=None,
    sampler:str='grid', ksize:int=16, n_features:int=196,
    force_hub_reload:bool=False
):
    assert model_name in valid_models, f'Invalid model name: {model_name}. Valid models are: {valid_models}'
    spotpaths = dict(
        spot_mae_b16='checkpoints/spot_mae_b16.pth',
        spot_21k_b16='checkpoints/spot_21k_b16.pth',
        spot_1k_b16='checkpoints/spot_1k_b16.pth',
    )
    
    timm_names = dict(
        vit_21k_b16='vit_base_patch16_224.augreg2_in21k_ft_in1k',
        vit_1k_b16='vit_base_patch16_224.augreg_in1k',
    )
    use_timm = False
    use_mae = False
    if model_name not in spotpaths:
        if 'mae' not in model_name:
            use_timm = True
        else:
            use_mae = True

    if use_timm:
        model = timm.create_model(timm_names[model_name], pretrained=True)
        if n_features < 196:
            model.patch_drop = patch_dropout.PatchDropout(1-n_features/196)
        return model.eval()
    elif use_mae:
        model = torch.hub.load(
            'mariuaas/mae', 
            'mae_vit_base_patch16_in1k',
            pretrained=True,
            global_pool=True,
            source='github',
            force_reload=force_hub_reload
        )
        if n_features < 196:
            model.patch_drop = patch_dropout.PatchDropout(1-n_features/196) # type: ignore
        return model.eval() # type: ignore
    
    kwargs = {'depth': 12, 'embed_dim': 768, 'heads': 12, 'dop_path': 0.0}
    kwargs['ksize'] = ksize
    kwargs['n_features'] = n_features
    kwargs['n_classes'] = 1000
    kwargs['sampler'] = sampler

    specifics = dict(
        spot_21k_b16=dict(
            qkv_bias=True,
            learnable_logprior=False,
            logprior=None,
            lnqk=False,
            pre_norm=False,
        ),
        spot_1k_b16=dict(
            qkv_bias=True,
            learnable_logprior=False,
            logprior=None,
            lnqk=False,
            pre_norm=False,
        ),
        spot_mae_b16=dict(
            qkv_bias=True,
            learnable_logprior=False,
            logprior=None,
            lnqk=False,
            pre_norm=False,
            global_pool=True,
        ),
    )[model_name]
    kwargs = {**kwargs, **specifics}
    model = nn.SPoTClassifier(**kwargs)
    
    path = path if path is not None else spotpaths[model_name]
    state_dict = torch.load(path, map_location='cpu', weights_only=False)
    
    if 'model' in state_dict:
        state_dict = state_dict['model']    
    
    if 'tokenizer.logprior' in state_dict:
        del state_dict['tokenizer.logprior']
    
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def get_validation_transform(model_name:str, imgsize:int = 224, crop_pct:float = 0.875):
    assert model_name in valid_models, f'Invalid model name: {model_name}. Valid models are: {valid_models}'
    preimgsize = int(round(imgsize / crop_pct))
    use_standard_transform = False
    if 'spot' in model_name:
        use_standard_transform = True
    elif 'mae' in model_name: 
        use_standard_transform = True

    if use_standard_transform:
        normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    else:
        normalize = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    return v2.Compose([
        v2.RandomResizedCrop((preimgsize, preimgsize), (1,1), interpolation=3),
        v2.CenterCrop((imgsize, imgsize)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        normalize
    ])


def create_validation(
    model_name:str, sampler:str='grid_uniform', n_features:int=196, ksize:int=16, 
    modelpath:str|None=None, dataset:str='IN1k', datapath:str='data', 
    override_extensions:list[str]=['jpg', 'cls', '_name'], 
    disable_model_gradients:bool=True, num_workers:int=4, batch_size:int=256, 
    crop_pct:float=0.875, prefetch_factor:int=2, pin_memory:bool=True, 
):
    '''Create validation, model dataset and dataloader.

    Parameters
    ----------
    model_name : str
        Name of the model to load.
    sampler : str, optional
        Sampler to use for the model, by default 'grid_uniform'.
    n_features : int, optional
        Number of features to use in the model, by default 196.
    ksize : int, optional
        Kernel size to use in the model, by default 16.
    modelpath : str, optional
        Path to the model weights, by default None.
    dataset : str, optional
        Dataset name, by default 'IN1k'
    datapath : str, optional
        Path to the dataset, by default 'data'.
    override_extensions : list[str], optional
        List of extensions to override in the dataset, by default ['jpg', 'cls', '_name'].
    disable_model_gradients : bool, optional
        Whether to disable gradients for the model, by default True.
    num_workers : int, optional
        Number of workers to use for the dataloader, by default 4.
    batch_size : int, optional
        Batch size to use for the dataloader, by default 256.
    crop_pct : float, optional
        Crop percentage to use for the validation transform, by default 0.875.
    prefetch_factor : int, optional
        Number of batches to prefetch, by default 2.
    pin_memory : bool, optional
        Whether to pin memory for the dataloader, by default True.

    Returns
    -------
    model : torch.nn.Module
        The loaded model.
    data : QuixDataset
        The dataset used for validation.
    dataloader : torch.utils.data.DataLoader
        The dataloader for the validation dataset.
    '''
    assert model_name in valid_models, f'Invalid model name: {model_name}. Valid models are: {valid_models}'
    tf = (
        get_validation_transform(model_name, imgsize=224, crop_pct=crop_pct),
        Identity(),
        Identity(),
    )
    data = QuixDataset(
        dataset, datapath, False, override_extensions=override_extensions
    ).map_tuple(*tf) # type: ignore
    dataloader = torch.utils.data.DataLoader(
        data, batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory,
    )
    model = load_trained_model(model_name, modelpath, sampler, ksize, n_features)
    model.eval()
    if disable_model_gradients:
        for param in model.parameters():
            param.requires_grad = False
    return model, data, dataloader        

    
