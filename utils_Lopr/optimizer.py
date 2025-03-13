import torch
import torch.nn as nn

def get_optimizer(net, lr):
        """
        Build optimizer, set weight decay of normalization to 0 by default.
        """
        def check_keywords_in_name(name, keywords=()):
            isin = False
            for keyword in keywords:
                if keyword in name:
                    isin = True
            return isin

        def set_weight_decay(model, skip_list=(), skip_keywords=()):
            has_decay = []
            no_decay = []

            for name, param in model.named_parameters():
                # check what will happen if we do not set no_weight_decay
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                        check_keywords_in_name(name, skip_keywords):
                    no_decay.append(param)
                    # print(f"{name} has no weight decay")
                else:
                    has_decay.append(param)
            return [{
                'params': has_decay
            }, {
                'params': no_decay,
                'weight_decay': 0.
            }]

        skip = {}
        skip_keywords = {}
        if hasattr(net, 'no_weight_decay'):
            skip = net.no_weight_decay()
        if hasattr(net, 'no_weight_decay_keywords'):
            skip_keywords = net.no_weight_decay_keywords()
        parameters = set_weight_decay(net, skip, skip_keywords)

        optimizer = torch.optim.AdamW(parameters,
                                        eps=1e-8,
                                        betas=(0.9, 0.999),
                                        lr=lr,
                                        weight_decay=1e-4)

        return optimizer


def get_lr_scheduler(optimizer):

    
    
    from utils_Lopr.custom_scheduler import CustomScheduler
    lr_scheduler = CustomScheduler(
        optimizer=optimizer,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=10,
        lr_decay_steps=500,
        lr_decay_style='cosine',
        start_wd=1e-4,
        end_wd=1e-4,
        wd_incr_style='constant',
        wd_incr_steps=500
    )
    return lr_scheduler
    
    