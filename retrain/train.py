"""Training the model."""

import copy
from datetime import datetime # pylint: disable=syntax-error
import json
import os
import random # pylint: disable=syntax-error
import time

from easydict import EasyDict as edict # pylint: disable=import-error
from tqdm import tqdm
import wandb # pylint: disable=syntax-error, import-error
import yaml
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch # pylint: disable=wrong-import-position
from torch import nn # pylint: disable=wrong-import-position
import transformers as tf # pylint: disable=wrong-import-position

import dataloader # pylint: disable=wrong-import-position, no-name-in-module
from prune_utils import PruneInitialize # pylint: disable=wrong-import-position, wrong-import-order
import params

timeprint = lambda x: print(str(datetime.now()), x)  # pylint: disable=C3001

LOG_FREQUENCY = 50
MODEL_CKPT_FREQUENCY = 2000
SEED = 432

torch.manual_seed(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(SEED)

scaler = torch.cuda.amp.GradScaler()


def train_step(model, batch, subbatch_size, loss_fn, optimizer, device, teacher_model):
    """A single training step."""
    model.train()
    correct, total_tokens, teacher_correct, total_loss = 0, 0, 0, 0

    input_ids = batch["input_ids"]
    label_ids = input_ids[:, 1:].contiguous().to(device)
    input_ids = input_ids[:, :-1].contiguous().to(device)
    attention_mask = batch["attention_mask"][:, 1:].to(device)

    with torch.cuda.amp.autocast():
        # Split into subbatches
        for i in range(0, input_ids.shape[0], subbatch_size):
            subbatch_input_ids = input_ids[i:i + subbatch_size, :]
            subbatch_label_ids = label_ids[i:i + subbatch_size, :]
            subbatch_attention_mask = attention_mask[i:i + subbatch_size, :]

            logits = model(input_ids=subbatch_input_ids).logits
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(
                        input_ids=subbatch_input_ids.to(teacher_model.device)).logits
                    teacher_logits = teacher_logits.detach().to(logits.device)
                loss = loss_fn(logits, subbatch_label_ids, subbatch_attention_mask, teacher_logits)
                teacher_predicts = teacher_logits.argmax(-1)
                teacher_correct += ((
                    teacher_predicts == subbatch_label_ids).float() * subbatch_attention_mask).sum()
            else:
                loss = loss_fn(logits, subbatch_label_ids, subbatch_attention_mask)

            scaler.scale(loss).backward()
            total_loss += loss.item()

            # Calculate accuracy
            predicts = logits.argmax(-1)
            correct += ((predicts == subbatch_label_ids).float() * subbatch_attention_mask).sum()
            total_tokens += subbatch_attention_mask.sum()

    scaler.step(optimizer)#.step()
    optimizer.zero_grad()
    model.zero_grad()
    scaler.update()

    accuracy = correct / total_tokens

    if teacher_model is not None:
        teacher_accuracy = teacher_correct / total_tokens
        return {"loss": total_loss, "accuracy": 100 * accuracy.item(),
                "teacher_accuracy": 100 * teacher_accuracy.item()}

    return {"loss": total_loss, "accuracy": 100 * accuracy.item()}


def evaluate_model(model, dataiterator, subbatch_size, loss_fn, device, use_wandb):
    """Evaluate the model."""
    model.eval()

    total_loss = 0
    total_accuracy = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataiterator):
            input_ids = batch["input_ids"]
            label_ids = input_ids[:, 1:].contiguous().to(device)
            input_ids = input_ids[:, :-1].contiguous().to(device)
            attention_mask = batch["attention_mask"][:, 1:].to(device)

            # Split into subbatches
            for i in range(0, input_ids.shape[0], subbatch_size):
                subbatch_input_ids = input_ids[i:i + subbatch_size, :]
                subbatch_label_ids = label_ids[i:i + subbatch_size, :]
                subbatch_attention_mask = attention_mask[i:i + subbatch_size, :]

                subbatch_logits = model(input_ids=subbatch_input_ids).logits
                subbatch_loss = loss_fn(
                    subbatch_logits, subbatch_label_ids, subbatch_attention_mask)

                # Calculate accuracy
                subbatch_predicts = subbatch_logits.argmax(-1)
                subbatch_correct = (
                    subbatch_predicts == subbatch_label_ids).float() * subbatch_attention_mask
                subbatch_accuracy = subbatch_correct.sum() / subbatch_attention_mask.sum()

                total_loss += subbatch_loss.item() * subbatch_attention_mask.sum()
                total_accuracy += subbatch_accuracy.item() * subbatch_attention_mask.sum()
                total_tokens += subbatch_attention_mask.sum()

    # Print and Log to wandb
    if use_wandb:
        wandb.log({"eval_loss": total_loss / total_tokens,
                  "eval_accuracy": 100 * total_accuracy / total_tokens})
    print("Eval Loss: ", total_loss / total_tokens)
    print("Eval Accuracy: ", 100 * total_accuracy / total_tokens)
    return


def save_checkpoint(iter_idx, model, optimizer, path, device):
    """Save model checkpoint, while moving them to CPU. Returns time taken to save."""
    start = time.time()
    optimizer_to(optimizer, 'cpu')

    if isinstance(model, nn.DataParallel):
        torch.save(
            {
                'iter_idx': iter_idx,
                'model_state_dict': model.cpu().module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
    else:
        torch.save(
            {
                'iter_idx': iter_idx,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)

    # Move to Device
    model = model.to(device)
    optimizer_to(optimizer, device)

    timeprint(f"Model saved at epoch {iter_idx}")
    return time.time() - start


def load_checkpoint(path):
    """Load the model checkpoint with highest iterations cnt."""
    if not os.path.isfile(path):
        possible_ckpt_paths = []
        for single_filename in os.listdir(path):
            if single_filename.startswith('ckpt_'):
                numerical_path = single_filename.lstrip('ckpt_')
                if all(char in '0123456789'
                       for char in numerical_path.split('.')[0]):
                    possible_ckpt_paths.append(single_filename)
        print(os.listdir(path))
        if not possible_ckpt_paths:
            raise ValueError(f"No saved checkpoint at {path}")

        latest_ckpt_file = sorted(
            possible_ckpt_paths,
            key=lambda x: int(x.lstrip('ckpt_').split('.')[0]))[-1]
        path = os.path.join(path, latest_ckpt_file)
    print(path)
    checkpoint = torch.load(path)
    if ['iter_idx'] in checkpoint:
        return (checkpoint['iter_idx'],
                checkpoint['model_state_dict'],
                checkpoint['optimizer_state_dict']), path
    return (0, checkpoint['model_state_dict'], None), path


def optimizer_to(optim, device):
    """Move Optimizer to CPU/GPU."""
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:  # pylint: disable=W0212
                param._grad.data = param._grad.data.to(device)  # pylint: disable=W0212
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:  # pylint: disable=W0212
                        subparam._grad.data = subparam._grad.data.to(device)  # pylint: disable=W0212


def log_metrics(logs, times, use_wandb, iter_idx, generated_samples):
    """Log the metrics and optionally to wandb."""
    print()
    timeprint(f"Step {iter_idx}")

    for log_name, log_value in logs.items():
        print(f"{log_name}: {log_value:.4f}", end=" | ")

    total_time = time.time() - times['Start']
    print(f"Time      ={round(total_time, 2)}sec", end=" ")
    print(f"(Train {round(100 * times['Train_step']/total_time, 1)}", end="% ")
    print(f"Save {round(100 * times['Save']/total_time, 1)}%)")
    # _ = [print(x) for x in generated_samples]

    if use_wandb:
        wandb_dict = {}
        for log_name, log_value in logs.items():
            wandb_dict[log_name] = log_value

        wandb_dict["Train Time"] = 100 * times['Train_step'] / total_time
        wandb_dict["Save Time"] = 100 * times['Save'] / total_time
        wandb_dict["Examples"] = generated_samples
        wandb.log(wandb_dict, step=iter_idx)

    for key in logs:
        logs[key] = 0

    # print("Generated Samples:", generated_samples)


def main(args):
    """Initializes Dataloader, Model, Optimizer and then trains the model."""
    # Prepeare save folder and store model config
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, str(len(os.listdir(args.save_dir))))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    json.dump(vars(args), open(os.path.join(args.save_dir, "params.json"), "w+")) # pylint: disable=unspecified-encoding
    timeprint(f"Save directory is {args.save_dir}")

    # Set Model, loss_fn and optimizer.
    model = tf.AutoModelForCausalLM.from_pretrained(args.model_card)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def loss_fn_eval(logits, label_ids, pad_mask):
        losses = ce_loss(logits.permute(0, 2, 1), label_ids)
        return (losses * pad_mask).sum() / pad_mask.sum()

    timeprint("Model, Loss Fn and Optimizer classes set.")

    # Load saved model, optimizer if any.
    iter_idx, prev_iter_idx = 0, 0
    if args.load_dir.strip() != "":
        (prev_iter_idx, model_state_dict, optimizer_state_dict) = load_checkpoint(
            args.load_dir)[0]
        model.load_state_dict(model_state_dict)
        if optimizer_state_dict is not None and optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)
    iter_idx = prev_iter_idx
    print("iter_idx", iter_idx)

    # Move to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # pylint: disable=no-member
    model = model.to(device)
    if device.type != "cpu":
        model = torch.nn.DataParallel(model)
    if args.wandb:
        wandb.watch(model)
    optimizer_to(optimizer, 'cuda')

    # If Knowledge Distillation is used, load the teacher model.
    if args.do_knowledge_distillation:
        # TODO: Try loss matching at each layer.
        if args.kd_teacher_model_path is not None:
            teacher_model = tf.AutoModelForCausalLM.from_pretrained(args.model_card)
            if args.kd_teacher_model_path.strip() != "":
                teacher_model.load_state_dict(load_checkpoint(args.kd_teacher_model_path)[0][3])
            if torch.cuda.device_count() > 1:
                teacher_device = f"cuda:{torch.cuda.device_count() - 1}"
            else:
                teacher_device = "cpu"
            teacher_model = teacher_model.to(teacher_device)
        else:
            teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        kd_loss = nn.KLDivLoss(reduction='none')
        def loss_fn(logits, label_ids, pad_mask, teacher_logits):
            # Compute losses againts target labels
            losses = ce_loss(
                logits.permute(0, 2, 1), label_ids)
            losses = (losses * pad_mask).sum() / pad_mask.sum()

            kd_losses = kd_loss(
                nn.functional.log_softmax(logits / args.kd_temperature, dim=-1),
                nn.functional.softmax(teacher_logits / args.kd_temperature, dim=-1))
            kd_losses = (kd_losses * pad_mask.unsqueeze(-1)).sum() / pad_mask.sum()

            return losses + args.kd_weight * kd_losses
        logs = {key: 0 for key in ["Loss", "Accuracy", "Teacher_Accuracy"]}
    else:
        teacher_model = None
        loss_fn = loss_fn_eval
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        logs = {key: 0 for key in ["Loss", "Accuracy"]}

    # Create the pruner model class
    if args.do_prune:
        iterative_pruner = prune_utils.Pruner(args.prune_recipe, iter_steps=iter_idx)
        # Initialize the model with initial pruning.
        iterative_pruner.init_prune(model=model)
    else:
        iterative_pruner = prune_utils.DummyPruner()

    # Initialize log dicts for training loop.
    times = {'Train_step': 0.0, 'Save': 0.0, 'Start': time.time()}

    def update_logs(train_log):
        for key in logs:
            logs[key] += train_log[key.lower()]

    print("Starting training")
    # Create the dataloaders.
    train_loader = dataloader.TextFileDataset(
        args.train_data_path, args.model_card, args.max_len, eval_mode=False, dummy_mode=args.dummy)
    # eval_loader = dataloader.TextFileDataset(
    #     args.eval_data_path, args.model_card, args.max_len, eval_mode=True, dummy_mode=args.dummy)
    if parser_args.dummy:
        train_loader.data = train_loader.data[::len(train_loader.data)//500]

    timeprint("Data is loaded")

    train_dataiterator = torch.utils.data.DataLoader(
        dataset=train_loader, batch_size=args.batch_size, num_workers=1, shuffle=False,
        collate_fn=dataloader.collate)

    # # Run evaluation on the model before training.
    # eval_dataloader = torch.utils.data.DataLoader(
    #     dataset=eval_loader, batch_size=args.batch_size, num_workers=1,
    #     collate_fn=dataloader.collate)
    # evaluate_model(model, eval_dataloader, 2*args.subbatch_size, loss_fn_eval, device, args.wandb)

    for batch in tqdm(train_dataiterator):
        if iter_idx and iter_idx % LOG_FREQUENCY == 0:
            model.eval()
            gen_fn = model.generate if device.type == "cpu" else model.module.generate
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            predicts = gen_fn(
                input_ids=batch['input_ids'][:1, :-1])

            generated_samples = [x.tolist()[0] for x in [predicts, batch['input_ids']]]
            generated_samples = train_loader.tokenizer.batch_decode(
                generated_samples, skip_special_tokens=True)

        start = time.time()
        train_logs = train_step(
            model, batch, args.subbatch_size, loss_fn, optimizer, device, teacher_model)
        times["Train_step"] += (time.time() - start)
        update_logs(train_logs)

        if iter_idx and iter_idx != prev_iter_idx and iter_idx % LOG_FREQUENCY == 0:
            # Normalize logs.
            logs = {log_key: log_val/LOG_FREQUENCY for log_key, log_val in logs.items()}
            # Log to wandb.
            if iter_idx - prev_iter_idx > LOG_FREQUENCY + 1:
                log_metrics(logs, times, args.wandb, iter_idx, generated_samples)
            logs = {log_key: 0.0 for log_key, _ in logs.items()}

            torch.cuda.empty_cache()

        save_path = os.path.join(
            args.save_dir, f"ckpt_{iter_idx}.pt")
            iterative_pruner.remove_prune(model=model)
            times["Save"] += save_checkpoint(
                iter_idx, model, optimizer, save_path, device)
            iterative_pruner.init_prune(model=model)

        iter_idx += 1



if __name__ == '__main__':
    # Read parameters from command line and then load config.
    parser_args = params.parse_arguments()

    # Setup wandb.
    if parser_args.wandb:
        wandb.init(project="sparsify", name=parser_args.model_card, config=parser_args)

    # Prepare path to datasets.
    main(parser_args)
