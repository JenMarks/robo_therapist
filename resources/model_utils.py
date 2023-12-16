import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import AdamW

from resources.data_utils import PadCollate
from resources.general_utils import (get_linear_schedule_with_warmup,
                                     gpu_information_summary,
                                     set_seed)


def train(train_dataset,
          model,
          tokenizer,
          per_gpu_train_batch_size,
          learning_rate,
          num_train_epochs,
          pad_values,
          evaluate_during_training=False,
          valid_dataset=None,
          max_steps=-1,
          gradient_accumulation_steps=1,
          weight_decay=0,
          adam_epsilon=1e-8,
          warmup_steps=0,
          max_grad_norm=1,
          fp16=False,
          fp16_opt_level="O1",
          seed_value=93,
          logging_steps=1,
          ):
    tb_writer = SummaryWriter()
    n_gpu, device = gpu_information_summary()
    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=PadCollate(pad_values=pad_values)
    )

    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // len(train_dataloader) // gradient_accumulation_steps + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    model.resize_token_embeddings(len(tokenizer))
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    if fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    if n_gpu > 0:
        model = torch.nn.DataParallel(model)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per GPU = {per_gpu_train_batch_size}")
    print(f"  Total train batch size (w. parallel  accumulation) = {train_batch_size * gradient_accumulation_steps}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {t_total}")

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    set_seed(seed_value=seed_value, n_gpu=n_gpu)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            labels = batch[0]
            labels = labels.to(device)

            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            model.train()
            outputs = model(**inputs, labels=labels)
            loss = outputs[0]

            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if logging_steps > 0 and global_step % logging_steps == 0:
                    if evaluate_during_training:
                        results = evaluate(model=model,
                                           dataset=valid_dataset,
                                           pad_values=pad_values,
                                           batch_size=per_gpu_train_batch_size)
                        for key, value in results.items():
                            tb_writer.add_scalar(key, value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss/training", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

            if 0 < max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < max_steps < global_step:
            train_iterator.close()
            break
    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(model, dataset, pad_values, batch_size):
    n_gpu, device = gpu_information_summary(show=False)
    eval_batch_size = batch_size * max(1, n_gpu)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size,
                                 collate_fn=PadCollate(pad_values=pad_values))

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    print("***** Running evaluation *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Batch size = {eval_batch_size}")

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        labels = batch[0]
        labels = labels.to(device)

        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
        }
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result = {"perplexity": perplexity, "loss/validation": eval_loss}
    return result


def generate(text,
             tokenizer,
             model,
             stop_token=".",
             length=30,
             num_return_sequences=1,
             temperature=1,
             k=50,
             p=0.95
             ):
    n_gpu, device = gpu_information_summary(show=False)
    model.to(device)

    encoded_prompt = tokenizer.encode(text,
                                      add_special_tokens=False,
                                      return_tensors="pt",
                                      add_space_before_punct_symbol=True
                                      )
    encoded_prompt = encoded_prompt.to(device)
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=length,
        temperature=temperature,
        top_k=k,
        top_p=p,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    generated_sequences = []
    for generated_sequence in output_sequences:
        generated_sequence = generated_sequence.tolist()
        in_out_text = tokenizer.decode(generated_sequence,
                                       clean_up_tokenization_spaces=True,
                                       skip_special_tokens=True)
        out_text = in_out_text[len(text)-1:]
        out_text = out_text[: out_text.find(stop_token) if stop_token else None]
        generated_sequences.append([text, out_text])

    return generated_sequences
