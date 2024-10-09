from .dataset import DefaultDataset
from .utils import load_model_and_tokenizer

import transformers


def finetune(
    model_dir: str,
    data_file: str,
    out_dir: str,
    epochs: int = 5,
    per_device_batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None
):
    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    dataset = DefaultDataset(
        data_file,
        tokenizer=tokenizer,
        max_len=max_len
    )

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        bf16=True,
        report_to='none'        # Disable wandb
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn()
    )

    model.config.use_cache = False  # silence the warnings.
    trainer.train()
    trainer.save_model(out_dir)
