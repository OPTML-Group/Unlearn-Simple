import os
from datasets import load_dataset
from utils import write_json, write_text

os.makedirs('data', exist_ok=True)

for corpus, Corpus in zip(['news', 'books'], ['News', 'Books']):
    for split in ['forget_qa', 'retain_qa', 'forget_qa_icl', 'retain_qa_icl']:
        data = load_dataset(f"muse-bench/MUSE-{Corpus}", 'knowmem', split=split)
        questions, answers = data['question'], data['answer']
        knowmem = [
            {'question': question, 'answer': answer}
            for question, answer in zip(questions, answers)
        ]
        full_path = os.path.abspath(f"data/{corpus}/knowmem/{split}.json")
        write_json(knowmem, full_path)
        print(f"Saved {len(knowmem)} knowmem entries to {full_path}")

    for split in ['forget']:
        data = load_dataset(f"muse-bench/MUSE-{Corpus}", 'verbmem', split='forget')
        prompts, gts = data['prompt'], data['gt']
        verbmem = [
            {'prompt': prompt, 'gt': gt}
            for prompt, gt in zip(prompts, gts)
        ]
        full_path = os.path.abspath(f"data/{corpus}/verbmem/{split}.json")
        write_json(verbmem, full_path)
        print(f"Saved {len(verbmem)} verbmem entries to {full_path}")

    for split in ['forget', 'retain', 'holdout']:
        privleak = load_dataset(f"muse-bench/MUSE-{Corpus}", 'privleak', split=split)['text']
        privleak = list(privleak)
        path = f"data/{corpus}/privleak/{split}.json"
        full_path = os.path.abspath(path)
        write_json(privleak, full_path)
        print(f"Saved {len(privleak)} privleak entries to {full_path}")

    for split in ['forget', 'holdout', 'retain1', 'retain2']:
        raw = load_dataset(f"muse-bench/MUSE-{Corpus}", 'raw', split=split)['text']
        raw = list(raw)
        full_path_json = os.path.abspath(f"data/{corpus}/raw/{split}.json")
        write_json(raw, full_path_json)
        print(f"Saved {len(raw)} raw entries to {full_path_json}")
        full_path_txt = os.path.abspath(f"data/{corpus}/raw/{split}.txt")
        write_text("\n\n".join(raw), full_path_txt)
        print(f"Saved {len(raw)} raw entries to {full_path_txt}")


for crit in ['scal', 'sust']:
    for fold in range(1, 5):
        data = load_dataset(f"muse-bench/MUSE-News", crit, split=f"forget_{fold}")['text']
        data = list(data)  # Convert to list
        full_path_json = os.path.abspath(f"data/news/{crit}/forget_{fold}.json")
        write_json(data, full_path_json)
        full_path_txt = os.path.abspath(f"data/news/{crit}/forget_{fold}.txt")
        write_text("\n\n".join(data), full_path_txt)
        print(f"Saved {len(data)} {crit} entries to {full_path_txt}")
