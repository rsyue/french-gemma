"""
French Gemma 3 Training and Evaluation Loop.

This module defines the Pretrainer class which handles the optimization loop,
mixed-precision training, gradient accumulation, model evaluation, checkpoints
management (perplexity and loss based), and TensorBoard metrics logging.
"""
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

def generate_text(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int = 30,
    device: str = "cpu"
) -> str:
    """
    Autoregressively generates text from a prompt using greedy decoding.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    
    # If input is empty, seed with BOS
    if input_ids_tensor.shape[1] == 0 and bos_id is not None:
        input_ids_tensor = torch.tensor([[bos_id]], dtype=torch.long).to(device)
        
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids_tensor)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids_tensor = torch.cat([input_ids_tensor, next_token], dim=-1)
            
            if eos_id is not None and next_token.item() == eos_id:
                break
                
    decoded = tokenizer.decode(input_ids_tensor[0].tolist(), skip_special_tokens=True)
    if isinstance(decoded, list):
        return decoded[0] if decoded else ""
    return str(decoded)

class Pretrainer:
    """
    Drives pretraining, optimization, AMP, gradient clipping/accumulation,
    evaluation, TensorBoard logging, and best-checkpoint retention.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        freeze_manager: Any,
        device: str,
        amp_enabled: bool = True,
        amp_dtype: str = "bfloat16",
        grad_clip_norm: float = 1.0,
        grad_accum_steps: int = 1,
        log_interval: int = 50,
        eval_interval: int = 500,
        save_interval: int = 1000,
        output_dir: str = "./checkpoints",
        tb_log_dir: str = "./runs",
        max_eval_batches: Optional[int] = 20,
        max_checkpoints: int = 5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.freeze_manager = freeze_manager
        self.device = device
        self.amp_enabled = amp_enabled
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.max_eval_batches = max_eval_batches
        self.max_checkpoints = max_checkpoints
        self.periodic_checkpoints: List[str] = []
        
        # Setup AMP dtype
        if amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "float16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32
            
        self.device_type = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
        
        # GradScaler is needed only for CUDA + float16
        self.scaler = None
        if self.amp_enabled and self.device_type == "cuda" and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda")
            
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        # Top perplexity and training loss checkpoints
        self.best_ppl_checkpoints: List[Dict[str, Any]] = []
        self.best_checkpoints = self.best_ppl_checkpoints
        self.best_loss_checkpoints: List[Dict[str, Any]] = []
        self.latest_train_loss: float = float("inf")
        self.last_log_time = time.time()

    def train_epoch(self, epoch: int, global_step: int) -> int:
        """
        Runs one training epoch.
        """
        self.model.train()
        accum_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Update freeze manager layers if schedule matches
            if self.freeze_manager is not None:
                self.freeze_manager.step(global_step)
                
            # Move inputs to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
            labels = batch["labels"].to(self.device) if "labels" in batch else input_ids.clone()
            
            # Forward pass under AMP
            # torch.amp.autocast supports mps only on very new torch versions,
            # so fallback to no autocast on mps if error occurs
            try:
                autocast_context = torch.amp.autocast(
                    device_type=self.device_type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled
                )
            except Exception:
                # Fallback to cpu/cuda autocast or disabled autocast
                device_for_cast = "cuda" if self.device_type == "cuda" else "cpu"
                autocast_context = torch.amp.autocast(
                    device_type=device_for_cast,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled and device_for_cast == "cuda"
                )
                
            with autocast_context:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / self.grad_accum_steps
                
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accum_loss += loss.item() * self.grad_accum_steps
            
            # Optimizer step (respecting gradient accumulation steps)
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    
                global_step += 1
                
                # Logging to console & TensorBoard
                if global_step % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - self.last_log_time
                    has_bs = self.train_dataloader and hasattr(self.train_dataloader, "batch_size")
                    loader_bs = self.train_dataloader.batch_size
                    batch_size = (loader_bs if loader_bs is not None else 1) if has_bs else 1
                    batches_processed = self.grad_accum_steps * self.log_interval
                    seqs_processed = batches_processed * batch_size
                    throughput = seqs_processed / elapsed if elapsed > 0 else 0
                    msg = (
                        f"Epoch {epoch} | Step {global_step} | Train Loss: {accum_loss:.4f} | "
                        f"LR: {current_lr:.6e} | Speed: {throughput:.2f} seqs/sec "
                        f"({elapsed:.2f}s elapsed)"
                    )
                    logger.info(msg)
                    print(msg)
                    self.writer.add_scalar("Loss/train", accum_loss, global_step)
                    self.writer.add_scalar("LR/train", current_lr, global_step)
                    self.writer.add_scalar("Speed/train_seqs_per_sec", throughput, global_step)
                    self.last_log_time = time.time()
                    
                self.latest_train_loss = accum_loss
                
                # Periodic Evaluation
                if global_step % self.eval_interval == 0:
                    self.evaluate(global_step)
                    self.model.train()
                    
                # Periodic Checkpoint
                if global_step % self.save_interval == 0:
                    self.save_checkpoint(global_step, perplexity=float("inf"))
                    
                accum_loss = 0.0
                
        return global_step

    def evaluate(self, global_step: int) -> float:
        """
        Runs the evaluation loop: computes perplexity and logs sample generation.
        """
        if self.val_dataloader is None:
            logger.info("Validation dataloader not provided, skipping evaluation.")
            return float("inf")
            
        logger.info("Starting validation loop...")
        print("Starting validation loop...")
        t_eval_start = time.time()
        
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        total_val_batches = len(self.val_dataloader) if hasattr(self.val_dataloader, "__len__") else None
        if self.max_eval_batches is not None:
            if total_val_batches is not None:
                total_val_batches = min(total_val_batches, self.max_eval_batches)
            else:
                total_val_batches = self.max_eval_batches
                
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if self.max_eval_batches is not None and batch_idx >= self.max_eval_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
                labels = batch["labels"].to(self.device) if "labels" in batch else input_ids.clone()
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                total_batches += 1
                
                if total_val_batches is not None and total_val_batches >= 10:
                    if (batch_idx + 1) % max(1, total_val_batches // 5) == 0 or (batch_idx + 1) == total_val_batches:
                        logger.info(f"Eval progress: batch {batch_idx + 1}/{total_val_batches}")
                        print(f"Eval progress: batch {batch_idx + 1}/{total_val_batches}")
                
        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        t_eval = time.time() - t_eval_start
        
        msg = (
            f"--- Eval Step {global_step} | Loss: {avg_loss:.4f} | "
            f"Perplexity: {perplexity:.4f} | Time: {t_eval:.2f}s ---"
        )
        logger.info(msg)
        print(msg)
        
        # Test generation
        sample_prompt = "Le français est"
        generated = generate_text(self.model, self.tokenizer, sample_prompt, max_new_tokens=20, device=self.device)
        logger.info(f"Sample generation: '{generated}'")
        print(f"Sample generation: '{generated}'")
        
        # Log to tensorboard
        self.writer.add_scalar("Loss/val", avg_loss, global_step)
        self.writer.add_scalar("Perplexity/val", perplexity, global_step)
        self.writer.add_text("Generation/val", generated, global_step)
        
        # Manage best checkpoints
        self.save_best_perplexity_checkpoint(global_step, perplexity)
        if self.latest_train_loss != float("inf"):
            self.save_best_loss_checkpoint(global_step, self.latest_train_loss)
        
        return perplexity

    def save_best_perplexity_checkpoint(self, global_step: int, perplexity: float):
        """
        Saves a checkpoint and retains only the 2 best checkpoints based on perplexity.
        """
        checkpoint_name = f"checkpoint-step-{global_step}-ppl-{perplexity:.2f}"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        
        should_save = False
        if len(self.best_ppl_checkpoints) < 2:
            should_save = True
        else:
            worst = max(self.best_ppl_checkpoints, key=lambda x: x["perplexity"])
            if perplexity < worst["perplexity"]:
                should_save = True
                worst_path = worst["path"]
                if os.path.exists(worst_path):
                    try:
                        import shutil
                        if os.path.isdir(worst_path):
                            shutil.rmtree(worst_path)
                        else:
                            os.remove(worst_path)
                        logger.info(f"Removed worst perplexity checkpoint file: {worst_path}")
                        print(f"Removed worst perplexity checkpoint file: {worst_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete checkpoint {worst_path}: {e}")
                self.best_ppl_checkpoints.remove(worst)
                
        if should_save:
            torch.save({
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "perplexity": perplexity
            }, checkpoint_path)
            self.best_ppl_checkpoints.append({"path": checkpoint_path, "perplexity": perplexity})
            logger.info(f"Saved new best perplexity checkpoint: {checkpoint_path}")
            print(f"Saved new best perplexity checkpoint: {checkpoint_path}")
        else:
            logger.info(f"No improvement in perplexity (current: {perplexity:.4f}). Leaving checkpoints in place.")
            print(f"No improvement in perplexity (current: {perplexity:.4f}). Leaving checkpoints in place.")

    def save_best_loss_checkpoint(self, global_step: int, train_loss: float):
        """
        Saves a checkpoint and retains only the 3 best checkpoints based on training loss.
        """
        checkpoint_name = f"checkpoint-step-{global_step}-loss-{train_loss:.4f}"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        
        should_save = False
        if len(self.best_loss_checkpoints) < 3:
            should_save = True
        else:
            worst = max(self.best_loss_checkpoints, key=lambda x: x["loss"])
            if train_loss < worst["loss"]:
                should_save = True
                worst_path = worst["path"]
                if os.path.exists(worst_path):
                    try:
                        import shutil
                        if os.path.isdir(worst_path):
                            shutil.rmtree(worst_path)
                        else:
                            os.remove(worst_path)
                        logger.info(f"Removed worst training loss checkpoint file: {worst_path}")
                        print(f"Removed worst training loss checkpoint file: {worst_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete checkpoint {worst_path}: {e}")
                self.best_loss_checkpoints.remove(worst)
                
        if should_save:
            torch.save({
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": train_loss
            }, checkpoint_path)
            self.best_loss_checkpoints.append({"path": checkpoint_path, "loss": train_loss})
            logger.info(f"Saved new best training loss checkpoint: {checkpoint_path}")
            print(f"Saved new best training loss checkpoint: {checkpoint_path}")
        else:
            logger.info(f"No improvement in training loss (current: {train_loss:.4f}). Leaving checkpoints in place.")
            print(f"No improvement in training loss (current: {train_loss:.4f}). Leaving checkpoints in place.")

    def save_best_checkpoint(self, global_step: int, perplexity: float):
        """
        Saves a checkpoint and retains only the 2 best checkpoints based on perplexity (backward compatible).
        """
        self.save_best_perplexity_checkpoint(global_step, perplexity)

    def save_checkpoint(self, global_step: int, perplexity: float = float("inf")):
        """
        Saves a standard step checkpoint.
        """
        checkpoint_name = f"checkpoint-step-{global_step}"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        torch.save({
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "perplexity": perplexity
        }, checkpoint_path)
        logger.info(f"Saved periodic checkpoint: {checkpoint_path}")
        print(f"Saved periodic checkpoint: {checkpoint_path}")
        
        self.periodic_checkpoints.append(checkpoint_path)
        if len(self.periodic_checkpoints) > self.max_checkpoints:
            oldest_path = self.periodic_checkpoints.pop(0)
            if os.path.exists(oldest_path):
                try:
                    import shutil
                    if os.path.isdir(oldest_path):
                        shutil.rmtree(oldest_path)
                    else:
                        os.remove(oldest_path)
                    logger.info(f"Deleted oldest periodic checkpoint: {oldest_path}")
                    print(f"Deleted oldest periodic checkpoint: {oldest_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete oldest checkpoint {oldest_path}: {e}")
