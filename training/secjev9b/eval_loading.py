from paths import WORK,SCRIPTS,SOURCE,BASE
"""Serialize model loading and preserve FP32 adapter weights on the BF16 backbone."""
import fcntl, gc, torch

def load_serial(checkpoint, device, options):
    from peft import load_peft_weights, set_peft_model_state_dict, get_peft_model_state_dict
    with open(str(WORK / 'evaluation-model-load.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tok, model = checkpoint.load(device, options)
        for name, parameter in model.lm.named_parameters():
            if 'lora_' in name:
                parameter.data = parameter.data.float()
        weights = load_peft_weights(checkpoint.path, device='cpu')
        have = set(get_peft_model_state_dict(model.lm))
        assert set(weights) == have
        set_peft_model_state_dict(model.lm, weights)
        loaded = get_peft_model_state_dict(model.lm)
        assert all((torch.equal(loaded[k].detach().cpu(), v.float()) for k, v in weights.items()))
        assert all((p.dtype == torch.float32 for n, p in model.lm.named_parameters() if 'lora_' in n))
        assert all((p.dtype == torch.float32 for p in model.head.parameters()))
        assert next((p for n, p in model.lm.named_parameters() if 'lora_' not in n)).dtype == torch.bfloat16
        del weights, loaded
        model.eval()
        gc.collect()
        return (tok, model)
