"""Score a native System One JSON request using a SecJev checkpoint."""
import argparse, json, pathlib, time, os, sys
sys.path.insert(0,str(pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()/'source/kev'))
import torch
from kev.api import SystemOneRequest, to_record, to_answers
from kev.checkpoint import Checkpoint, LoadOptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Local checkpoint directory or owner/repository@revision')
    parser.add_argument('--request', required=True, type=pathlib.Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--base', help='Optional local copy of the pinned Qwen backbone')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'bfloat16'], default='auto', help='auto: BF16 backbone for 9B, FP32 for earlier models; adapter/head stay FP32')
    args = parser.parse_args()
    request = SystemOneRequest.model_validate(json.loads(args.request.read_text()))
    checkpoint = Checkpoint(args.model)
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' or (args.dtype == 'auto' and '9b' in checkpoint.meta.base.lower()) else torch.float32
    if args.base:
        checkpoint.meta.base = args.base
        checkpoint.meta.base_revision = None
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    tokenizer, model = checkpoint.load(args.device, LoadOptions(dtype=dtype, merge=False, attn='sdpa'))
    if dtype == torch.bfloat16:
        from peft import load_peft_weights, set_peft_model_state_dict, get_peft_model_state_dict
        for name, parameter in model.lm.named_parameters():
            if 'lora_' in name:
                parameter.data = parameter.data.float()
        weights = load_peft_weights(checkpoint.path, device='cpu')
        assert set(weights) == set(get_peft_model_state_dict(model.lm))
        set_peft_model_state_dict(model.lm, weights)
        loaded = get_peft_model_state_dict(model.lm)
        assert all(torch.equal(loaded[k].detach().cpu(), v.float()) for k, v in weights.items())
        assert all(p.dtype == torch.float32 for n, p in model.lm.named_parameters() if 'lora_' in n)
        assert all(p.dtype == torch.float32 for p in model.head.parameters())
        del weights, loaded
    model.eval()
    record, metadata = to_record(request)
    encoded = model.encode(tokenizer, record, max_state=8192, max_branch=8192, strict=True)
    from kev.model import rows_of
    state, _, branches = rows_of(encoded)
    if any(len(state) + len(branch['ids']) > 8192 for branch in branches):
        raise ValueError('Each state + question must fit within 8192 tokens; inputs are never truncated.')
    if str(args.device).startswith('cuda'):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model.forward(encoded)
        probabilities = [torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
    if str(args.device).startswith('cuda'):
        torch.cuda.synchronize()
    result = {
        'model': args.model,
        'precision': ('BF16 backbone, FP32 LoRA/head' if dtype == torch.bfloat16 else 'FP32'),
        'answers': to_answers(probabilities, metadata),
        'probabilities': {m['id']: dict(zip(m['keys'], p)) for m, p in zip(metadata, probabilities)},
        'temperature': model.head.temperature,
        'forward_ms': (time.perf_counter() - started) * 1000,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
