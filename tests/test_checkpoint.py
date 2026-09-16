import pytest
import torch

from config import Config, DataConfig, ModelConfig, TrainingConfig
from data.loader import create_dataloader
from device import model_device
from model import GPT
from tokenizer import CharTokenizer
from training.checkpoint import load_checkpoint, load_training_checkpoint, save_checkpoint
from training.trainer import configure_optimizer, train

TEXT = "o gato está na casa"
CONFIG = ModelConfig(context_length=16, d_model=16, num_heads=2, num_layers=1, dropout=0.1)


def saved_model(tmp_path):
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(TEXT)
    model = GPT(CONFIG, tokenizer.vocab_size)
    path = tmp_path / "checkpoints" / "latest.pt"
    save_checkpoint(path, model, tokenizer)
    return model, tokenizer, path


def test_round_trip_preserves_config_vocab_and_outputs(tmp_path):
    model, tokenizer, path = saved_model(tmp_path)
    loaded_model, loaded_tokenizer = load_checkpoint(path)

    assert loaded_model.config == model.config
    assert loaded_tokenizer.vocab.id_to_token == tokenizer.vocab.id_to_token
    ids = torch.tensor([tokenizer.encode("o gato")])
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded_model(ids), model(ids))


def test_loaded_model_keeps_weight_tying(tmp_path):
    _, _, path = saved_model(tmp_path)
    loaded_model, _ = load_checkpoint(path)
    assert loaded_model.lm_head.proj.weight is loaded_model.embeddings.token_embedding.weight


def test_loaded_model_is_ready_for_inference(tmp_path):
    _, _, path = saved_model(tmp_path)
    loaded_model, _ = load_checkpoint(path)
    assert not loaded_model.training  # dropout desligado


# --- checkpoints de treino ---

TRAIN_MODEL = ModelConfig(context_length=8, d_model=16, num_heads=2, num_layers=1, dropout=0.1)
TRAINING = TrainingConfig(
    batch_size=4,
    learning_rate=1e-2,
    min_learning_rate=1e-3,
    warmup_steps=5,
    epochs=2,
    weight_decay=0.01,
    gradient_clip=1.0,
    seed=1,
    log_every=1000,
    checkpoint_every=10,
)
FULL_CONFIG = Config(DataConfig("corpus.txt", 0.1, 4), TRAIN_MODEL, TRAINING)
TOKENIZER = CharTokenizer.from_text(TEXT)


class Interrupted(Exception):
    pass


def make_loaders():
    ids = torch.tensor(TOKENIZER.encode(TEXT * 20))  # 93 janelas: 23 batches por época
    train_loader = create_dataloader(
        ids, context_length=8, stride=4, batch_size=4, shuffle=True, seed=TRAINING.seed
    )
    val_loader = create_dataloader(ids, context_length=8, stride=8, batch_size=4, shuffle=False)
    return train_loader, val_loader


AVAILABLE = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]
DEVICES = ["cpu"] + [name for name, available in AVAILABLE if available]


def new_run(device="cpu"):
    torch.manual_seed(0)
    model = GPT(TRAIN_MODEL, TOKENIZER.vocab_size).to(device)
    optimizer = configure_optimizer(model, TRAINING.learning_rate, TRAINING.weight_decay)
    return model, optimizer


def run(model, optimizer, *, progress=None, on_checkpoint=None, log=lambda line: None):
    train_loader, val_loader = make_loaders()
    return train(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=TRAINING.epochs,
        max_lr=TRAINING.learning_rate,
        min_lr=TRAINING.min_learning_rate,
        warmup_steps=TRAINING.warmup_steps,
        gradient_clip=TRAINING.gradient_clip,
        log_every=TRAINING.log_every,
        progress=progress,
        checkpoint_every=TRAINING.checkpoint_every,
        on_checkpoint=on_checkpoint,
        log=log,
    )


def interrupted_run(path, stop_at_step, device="cpu"):
    """Treina salvando cada checkpoint em `path` e 'cai' logo depois de salvar o passo pedido."""
    model, optimizer = new_run(device)

    def save_then_stop(progress):
        save_checkpoint(
            path, model, TOKENIZER, config=FULL_CONFIG, optimizer=optimizer, progress=progress
        )
        if progress.step == stop_at_step:
            raise Interrupted

    with pytest.raises(Interrupted):
        run(model, optimizer, on_checkpoint=save_then_stop)


def resume(path, *, restore_optimizer=True, device="cpu", log=lambda line: None):
    torch.manual_seed(123)  # outro estado aleatório (CPU e acelerador), como num processo novo
    checkpoint = load_training_checkpoint(path)
    model = checkpoint.model.to(device)
    optimizer = configure_optimizer(model, TRAINING.learning_rate, TRAINING.weight_decay)
    if restore_optimizer:
        optimizer.load_state_dict(checkpoint.optimizer_state)
    history = run(model, optimizer, progress=checkpoint.progress, log=log)
    return model, history


def test_checkpoints_are_offered_periodically_and_at_every_epoch_end():
    model, optimizer = new_run()
    seen = []
    run(model, optimizer, on_checkpoint=lambda p: seen.append((p.step, p.epoch, p.batches_done)))
    # (passo, época em andamento, batches já feitos nela); a época tem 23 batches.
    assert seen == [(10, 1, 10), (20, 1, 20), (23, 2, 0), (30, 2, 7), (40, 2, 17), (46, 3, 0)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stop_at_step", [23, 30])  # fim da época 1; meio da época 2
def test_resumed_training_matches_uninterrupted_training(tmp_path, stop_at_step, device):
    model, optimizer = new_run(device)
    history = run(model, optimizer)

    path = tmp_path / "latest.pt"
    interrupted_run(path, stop_at_step, device)
    resumed_model, resumed_history = resume(path, device=device)

    # Na CPU, ponto flutuante é determinístico: a retomada bate bit a bit. Num acelerador, a ordem
    # de redução dos kernels (matmul, attention) pode variar entre execuções, deixando uma
    # diferença na última casa do float32 (~1e-8); comparamos com uma tolerância mínima.
    exact = device == "cpu"
    for name, tensor in model.state_dict().items():
        resumed = resumed_model.state_dict()[name]
        if exact:
            assert torch.equal(resumed, tensor), name
        else:
            torch.testing.assert_close(resumed, tensor, rtol=0, atol=1e-6)
    for loss_name in ("train_loss", "val_loss"):
        resumed_losses = [getattr(r, loss_name) for r in resumed_history]
        original_losses = [getattr(r, loss_name) for r in history]
        if exact:
            assert resumed_losses == original_losses
        else:
            assert resumed_losses == pytest.approx(original_losses, rel=0, abs=1e-6)


def test_resuming_without_the_optimizer_state_changes_the_result(tmp_path):
    model, optimizer = new_run()
    run(model, optimizer)

    path = tmp_path / "latest.pt"
    interrupted_run(path, stop_at_step=30)
    resumed_model, _ = resume(path, restore_optimizer=False)

    weights = resumed_model.state_dict()
    assert any(not torch.equal(weights[name], t) for name, t in model.state_dict().items())


def test_training_checkpoint_restores_config_and_progress(tmp_path):
    path = tmp_path / "latest.pt"
    interrupted_run(path, stop_at_step=30)
    checkpoint = load_training_checkpoint(path)
    assert checkpoint.config == FULL_CONFIG
    assert (checkpoint.progress.step, checkpoint.progress.epoch) == (30, 2)
    assert checkpoint.progress.batches_done == len(checkpoint.progress.epoch_losses) == 7
    assert [r.epoch for r in checkpoint.progress.history] == [1]


def test_training_checkpoint_also_works_for_inference(tmp_path):
    path = tmp_path / "latest.pt"
    interrupted_run(path, stop_at_step=30)
    model, tokenizer = load_checkpoint(path)
    ids = torch.tensor([tokenizer.encode("o gato")])
    assert model(ids).shape == (1, 6, tokenizer.vocab_size)


@pytest.mark.skipif(len(DEVICES) == 1, reason="nenhum acelerador neste ambiente")
def test_accelerator_checkpoint_loads_on_cpu_and_warns_when_resumed_there(tmp_path):
    path = tmp_path / "latest.pt"
    interrupted_run(path, stop_at_step=30, device=DEVICES[1])

    model, _ = load_checkpoint(path)
    assert model_device(model).type == "cpu"

    logs = []
    resume(path, device="cpu", log=logs.append)
    assert any(line.startswith("aviso:") for line in logs)


def test_weights_only_checkpoint_cannot_resume_training(tmp_path):
    _, _, path = saved_model(tmp_path)
    with pytest.raises(ValueError):
        load_training_checkpoint(path)


def test_training_state_must_be_complete(tmp_path):
    model, optimizer = new_run()
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "x.pt", model, TOKENIZER, optimizer=optimizer)


def test_failed_save_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    model, tokenizer, path = saved_model(tmp_path)

    def broken_save(obj, file):
        open(file, "wb").write(b"metade de um arquivo")
        raise OSError("disco cheio")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        save_checkpoint(path, model, tokenizer)
    monkeypatch.undo()

    load_checkpoint(path)  # o arquivo anterior continua íntegro
    assert list(path.parent.iterdir()) == [path]  # e o temporário foi apagado
