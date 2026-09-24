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
    args = parser.parse_args()
    request = SystemOneRequest.model_validate(json.loads(args.request.read_text()))
    checkpoint = Checkpoint(args.model)
    if args.base:
        checkpoint.meta.base = args.base
        checkpoint.meta.base_revision = None
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    tokenizer, model = checkpoint.load(args.device, LoadOptions(dtype=torch.float32, merge=False, attn='sdpa'))
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
        'answers': to_answers(probabilities, metadata),
        'probabilities': {m['id']: dict(zip(m['keys'], p)) for m, p in zip(metadata, probabilities)},
        'temperature': model.head.temperature,
        'forward_ms': (time.perf_counter() - started) * 1000,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
