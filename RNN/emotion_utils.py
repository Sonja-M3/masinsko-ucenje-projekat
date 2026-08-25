"""Zajednicki alati za LSTM i GRU klasifikaciju emocija."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SEED = 7
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1
MIN_FREQ = 2
MAX_VOCAB_SIZE = 20_000
MAX_SEQUENCE_LENGTH = 80
BATCH_SIZE = 128
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
MODEL_CONFIG_KEYS = (
    "embedding_dim",
    "hidden_size",
    "num_layers",
    "dropout",
    "learning_rate",
)


def set_seed(seed: int = SEED) -> None:
    """Postavlja generatore slucajnih brojeva radi ponovljivosti."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Vraca CUDA uredjaj kada je dostupan, inace CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(text: str) -> list[str]:
    """Pretvara tekst u listu alfanumerickih tokena malim slovima."""
    return TOKEN_PATTERN.findall(text.lower())


def load_split(path: str | Path) -> tuple[list[str], list[str]]:
    """Ucitava skup u formatu ``tekst;oznaka``."""
    path = Path(path)
    texts: list[str] = []
    labels: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                text, label = line.rsplit(";", maxsplit=1)
            except ValueError as error:
                raise ValueError(
                    f"Neispravan red {line_number} u {path}: {line!r}"
                ) from error

            text = text.strip()
            label = label.strip()
            if not text or not label:
                raise ValueError(
                    f"Prazan tekst ili oznaka u redu {line_number} u {path}."
                )
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError(f"Skup podataka je prazan: {path}")
    return texts, labels


def build_vocabulary(
    texts: list[str],
    min_frequency: int = MIN_FREQ,
    max_vocabulary_size: int = MAX_VOCAB_SIZE,
) -> tuple[list[str], dict[str, int], Counter]:
    """Gradi recnik iskljucivo iz trening tekstova."""
    if max_vocabulary_size < 2:
        raise ValueError("Velicina recnika mora biti barem 2 zbog posebnih tokena.")

    token_counts: Counter = Counter()
    for text in texts:
        token_counts.update(tokenize(text))

    frequent_tokens = [
        token
        for token, frequency in token_counts.most_common()
        if frequency >= min_frequency
    ][: max_vocabulary_size - 2]

    index_to_token = [PAD_TOKEN, UNK_TOKEN, *frequent_tokens]
    token_to_index = {
        token: index for index, token in enumerate(index_to_token)
    }
    return index_to_token, token_to_index, token_counts


def build_label_mapping(
    train_labels: list[str], *other_label_splits: list[str]
) -> tuple[list[str], dict[str, int]]:
    """Gradi mapiranje klasa i proverava oznake ostalih skupova."""
    label_names = sorted(set(train_labels))
    label_to_index = {
        label: index for index, label in enumerate(label_names)
    }

    for labels in other_label_splits:
        unknown_labels = sorted(set(labels) - set(label_names))
        if unknown_labels:
            raise ValueError(
                f"Oznake koje ne postoje u trening skupu: {unknown_labels}"
            )
    return label_names, label_to_index


def numericalize(
    text: str,
    token_to_index: dict[str, int],
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> list[int]:
    """Pretvara tekst u ogranicenu sekvencu indeksa iz recnika."""
    token_ids = [
        token_to_index.get(token, UNK_IDX) for token in tokenize(text)
    ][:max_sequence_length]
    return token_ids if token_ids else [UNK_IDX]


class EmotionDataset(Dataset):
    """Dataset koji tekstove pretvara u tenzore tek pri pristupu primeru."""

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        token_to_index: dict[str, int],
        label_to_index: dict[str, int],
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("Broj tekstova i oznaka mora biti isti.")

        self.texts = texts
        self.labels = labels
        self.token_to_index = token_to_index
        self.label_to_index = label_to_index
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.tensor(
            numericalize(
                self.texts[index],
                self.token_to_index,
                self.max_sequence_length,
            ),
            dtype=torch.long,
        )
        label = torch.tensor(
            self.label_to_index[self.labels[index]], dtype=torch.long
        )
        return sequence, label


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    padding_index: int = PAD_IDX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dopunjava sekvence unutar jednog paketica do iste duzine."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor(
        [len(sequence) for sequence in sequences], dtype=torch.long
    )
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=padding_index
    )
    return padded_sequences, lengths, torch.stack(labels)


@dataclass
class EmotionData:
    train_texts: list[str]
    train_labels: list[str]
    val_texts: list[str]
    val_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]
    index_to_token: list[str]
    token_to_index: dict[str, int]
    token_counts: Counter
    label_names: list[str]
    label_to_index: dict[str, int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    max_sequence_length: int

    def encode(self, text: str) -> list[int]:
        return numericalize(
            text, self.token_to_index, self.max_sequence_length
        )


def prepare_emotion_data(
    data_dir: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
    min_frequency: int = MIN_FREQ,
    max_vocabulary_size: int = MAX_VOCAB_SIZE,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    seed: int = SEED,
) -> EmotionData:
    """Ucitava skupove i priprema zajednicke DataLoader objekte."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)

    train_texts, train_labels = load_split(data_dir / "train.txt")
    val_texts, val_labels = load_split(data_dir / "val.txt")
    test_texts, test_labels = load_split(data_dir / "test.txt")

    index_to_token, token_to_index, token_counts = build_vocabulary(
        train_texts,
        min_frequency=min_frequency,
        max_vocabulary_size=max_vocabulary_size,
    )
    label_names, label_to_index = build_label_mapping(
        train_labels, val_labels, test_labels
    )

    dataset_arguments = {
        "token_to_index": token_to_index,
        "label_to_index": label_to_index,
        "max_sequence_length": max_sequence_length,
    }
    train_dataset = EmotionDataset(
        train_texts, train_labels, **dataset_arguments
    )
    val_dataset = EmotionDataset(val_texts, val_labels, **dataset_arguments)
    test_dataset = EmotionDataset(
        test_texts, test_labels, **dataset_arguments
    )

    batch_collator = partial(collate_batch, padding_index=PAD_IDX)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=batch_collator,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=batch_collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=batch_collator,
        num_workers=0,
    )

    return EmotionData(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        index_to_token=index_to_token,
        token_to_index=token_to_index,
        token_counts=token_counts,
        label_names=label_names,
        label_to_index=label_to_index,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        max_sequence_length=max_sequence_length,
    )


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Izvrsava jednu trening ili evaluacionu epohu."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for token_ids, lengths, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(token_ids, lengths)
            loss = criterion(logits, labels)

            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            current_batch_size = labels.size(0)
            total_loss += loss.item() * current_batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += current_batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    learning_rate: float = 1e-3,
) -> dict[str, list[float]]:
    """Trenira model i na kraju vraca najbolje validaciono stanje."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: defaultdict[str, list[float]] = defaultdict(list)
    best_val_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(1, epochs + 1), desc="Epohe"):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        print(
            f"Epoha {epoch:02d} | "
            f"train loss: {train_loss:.4f}, train acc: {train_accuracy:.4f} | "
            f"val loss: {val_loss:.4f}, val acc: {val_accuracy:.4f}"
        )

    if best_state is None:
        raise ValueError("Broj epoha mora biti veci od nule.")
    model.load_state_dict(best_state)
    return dict(history)


def plot_history(history: dict[str, list[float]]) -> None:
    """Prikazuje gresku i tacnost kroz epohe."""
    epochs = range(1, len(history["train_loss"]) + 1)
    _, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Trening")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validacija")
    axes[0].set(
        title="Greška kroz epohe",
        xlabel="Epoha",
        ylabel="Cross-entropy loss",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        epochs, history["train_accuracy"], marker="o", label="Trening"
    )
    axes[1].plot(
        epochs, history["val_accuracy"], marker="o", label="Validacija"
    )
    axes[1].set(
        title="Tačnost kroz epohe",
        xlabel="Epoha",
        ylabel="Tačnost",
        ylim=(0, 1),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> tuple[float, float, list[int], list[int]]:
    """Racuna gresku, tacnost i predikcije nad jednim skupom."""
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_examples = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    with torch.no_grad():
        for token_ids, lengths, labels in tqdm(data_loader, desc="Evaluacija"):
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            logits = model(token_ids, lengths)
            loss = criterion(logits, labels)

            current_batch_size = labels.size(0)
            total_loss += loss.item() * current_batch_size
            total_examples += current_batch_size
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())

    average_loss = total_loss / total_examples
    accuracy = float(
        np.mean(np.asarray(true_labels) == np.asarray(predicted_labels))
    )
    return average_loss, accuracy, true_labels, predicted_labels


def print_evaluation(
    model: nn.Module,
    data_loader: DataLoader,
    label_names: list[str],
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Ispisuje izvestaj i prikazuje matricu konfuzije."""
    loss, accuracy, true_labels, predicted_labels = evaluate_model(
        model, data_loader, device
    )
    label_indices = list(range(len(label_names)))

    print(f"Test greška:  {loss:.4f}")
    print(f"Test tačnost: {accuracy:.4f}\n")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=label_indices,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=label_indices
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=label_names
    )
    _, axis = plt.subplots(figsize=(8, 7))
    display.plot(
        ax=axis, cmap="Blues", colorbar=False, values_format="d"
    )
    axis.set_title("Matrica konfuzije — test skup")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return loss, accuracy, true_labels, predicted_labels


def save_emotion_checkpoint(
    model: nn.Module,
    data: EmotionData,
    result: Mapping[str, Any],
    model_type: str,
    output_path: str | Path | None = None,
) -> Path:
    """Sacuvaj tezine modela, konfiguraciju, recnik i mapiranje klasa.

    Model mora vec biti istreniran. Funkcija kopira tenzore na CPU, pa se
    dobijeni checkpoint kasnije moze ucitati i na racunaru bez CUDA podrske.
    """
    normalized_model_type = model_type.strip().upper()
    if normalized_model_type not in {"GRU", "LSTM"}:
        raise ValueError("model_type mora biti 'GRU' ili 'LSTM'.")

    missing_keys = [key for key in MODEL_CONFIG_KEYS if key not in result]
    if missing_keys:
        raise KeyError(
            "Nedostaju parametri konfiguracije: " + ", ".join(missing_keys)
        )

    if output_path is None:
        models_dir = Path(__file__).resolve().parent.parent / "models" / "rnn"
        output_path = models_dir / f"{normalized_model_type.lower()}_emotion.pth"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": 1,
        "model_type": normalized_model_type,
        "model_name": result.get("name", normalized_model_type),
        "config": {key: result[key] for key in MODEL_CONFIG_KEYS},
        "training_result": dict(result),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "index_to_token": list(data.index_to_token),
        "token_to_index": dict(data.token_to_index),
        "label_names": list(data.label_names),
        "label_to_index": dict(data.label_to_index),
        "max_sequence_length": data.max_sequence_length,
        "padding_idx": PAD_IDX,
        "seed": SEED,
    }

    torch.save(checkpoint, output_path)
    return output_path
