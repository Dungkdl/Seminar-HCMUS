"""Hardware-aware throughput settings; optimizer batch and research protocol stay explicit."""
import os
import torch


def hardware():
    info = {'device': 'cpu', 'workers': min(8, os.cpu_count() or 1),
            'torch': str(torch.__version__), 'cuda_build': torch.version.cuda}
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        free, _ = torch.cuda.mem_get_info()
        info.update(device='cuda', name=prop.name, capability=list(torch.cuda.get_device_capability(0)),
                    total_gib=round(prop.total_memory / 2**30, 2), free_gib=round(free / 2**30, 2))
        # Check the installed build with a real kernel, including Blackwell support.
        try:
            x = torch.ones((16, 16), device='cuda')
            (x @ x).sum().item()
        except RuntimeError as exc:
            raise RuntimeError('CUDA smoke check failed. For RTX PRO 6000 Blackwell use a PyTorch '
                               'build supporting sm_120 (CUDA 12.8+); rebuild the bank on the matching image.') from exc
    return info


def feature_settings(config, info=None):
    info = hardware() if info is None else info
    large = info['device'] == 'cuda' and info.get('free_gib', 0) >= 70
    medium = info['device'] == 'cuda' and info.get('free_gib', 0) >= 30
    image = config.get('image_batch_size', 'auto')
    batches = ({'student': 256, 'teacher': 128, 'resnet50': 256, 'swin_b': 128} if large else
               {'student': 64, 'teacher': 32, 'resnet50': 128, 'swin_b': 32} if medium else
               {k: 8 for k in ('student', 'teacher', 'resnet50', 'swin_b')})
    if image != 'auto':
        if not isinstance(image, int) or isinstance(image, bool) or image < 1:
            raise ValueError('image_batch_size must be auto or a positive integer.')
        batches = {k: image for k in batches}
    workers = config.get('image_workers', 'auto')
    workers = info['workers'] if workers == 'auto' and info['device'] == 'cuda' else 0 if workers == 'auto' else workers
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError('image_workers must be auto or a nonnegative integer.')
    text = config.get('clinical_batch_size', 'auto')
    text = (128 if large else 32 if medium else 16) if text == 'auto' else text
    if not isinstance(text, int) or isinstance(text, bool) or text < 1:
        raise ValueError('clinical_batch_size must be auto or a positive integer.')
    return {'image_batches': batches, 'workers': workers, 'clinical_batch_size': text,
            'chunk_size': 2048, 'prefetch_factor': 2, 'hardware': info}
