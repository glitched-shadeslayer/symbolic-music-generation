import os
import torch
import torch.nn as nn
import torch.optim as optim
from music21 import converter, instrument, note, chord, stream
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "nottingham-dataset-master/MIDI/"  # Update to your MIDI folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MIDI Parser
# -----------------------------
def parse_midi_file(file_path):
    midi = converter.parse(file_path)
    parts = instrument.partitionByInstrument(midi)
    melody = []
    chords = []

    if parts: 
        part = parts.parts[0]
    else:
        part = midi.flat.notes

    for el in part.flat.notesAndRests:
        if isinstance(el, note.Note):
            melody.append(str(el.pitch))
            chords.append("C")  # Placeholder chord
        elif isinstance(el, chord.Chord):
            melody.append(str(el.root()))
            chords.append(str(el.commonName))
        elif isinstance(el, note.Rest):
            melody.append("Rest")
            chords.append("C")

    return melody, chords

# -----------------------------
# Load Dataset
# -----------------------------
all_melodies = []
all_chords = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".mid") or fname.endswith(".midi"):
        try:
            mel, chd = parse_midi_file(os.path.join(DATA_DIR, fname))
            if len(mel) > 0:
                all_melodies.append(mel)
                all_chords.append(chd)
        except Exception as e:
            print(f"Error parsing {fname}: {e}")

# -----------------------------
# Vocab
# -----------------------------
note_vocab = sorted(list({n for mel in all_melodies for n in mel}))
chord_vocab = sorted(list({c for chd in all_chords for c in chd}))

note_to_idx = {n: i for i, n in enumerate(note_vocab)}
note_to_idx["UNK"] = len(note_to_idx)
idx_to_note = {i: n for n, i in note_to_idx.items()}

chord_to_idx = {c: i for i, c in enumerate(chord_vocab)}
idx_to_chord = {i: c for c, i in chord_to_idx.items()}

# -----------------------------
# Prepare Data
# -----------------------------
x_data = []
y_data = []

for mel, chd in zip(all_melodies, all_chords):
    x_seq = [note_to_idx.get(n, note_to_idx["UNK"]) for n in mel]
    y_seq = [chord_to_idx[c] for c in chd]
    if len(x_seq) == len(y_seq):
        x_data.append(torch.tensor(x_seq))
        y_data.append(torch.tensor(y_seq))

# -----------------------------
# Train/Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# -----------------------------
# Model
# -----------------------------
class Harmonizer(nn.Module):
    def __init__(self, num_notes, num_chords, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_notes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_chords)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)

model = Harmonizer(len(note_to_idx), len(chord_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in zip(x_train, y_train):
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# -----------------------------
# Evaluation (Accuracy)
# -----------------------------
def evaluate(model, x_test, y_test):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in zip(x_test, y_test):
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            out = model(x)
            pred = out.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()

    acc = 100 * correct / total
    print(f"\nTest Accuracy: {acc:.2f}%")

evaluate(model, x_test, y_test)

# -----------------------------
# Detailed Classification Report
# -----------------------------
# def evaluate_with_metrics(model, x_test, y_test):
#     model.eval()
#     y_true = []
#     y_pred = []

#     with torch.no_grad():
#         for x, y in zip(x_test, y_test):
#             x = x.unsqueeze(0).to(device)
#             out = model(x)
#             preds = out.argmax(dim=-1).squeeze().cpu().tolist()
#             targets = y.squeeze().tolist()
#             y_pred.extend(preds)
#             y_true.extend(targets)

#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=chord_vocab))

def evaluate_with_metrics(model, x_test, y_test):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in zip(x_test, y_test):
            x = x.unsqueeze(0).to(device)
            out = model(x)
            preds = out.argmax(dim=-1).squeeze().cpu().tolist()
            targets = y.squeeze().tolist()

            # Ensure list format
            if isinstance(preds, int):
                preds = [preds]
            if isinstance(targets, int):
                targets = [targets]

            y_pred.extend(preds)
            y_true.extend(targets)

    # Compute only over classes actually present in y_true and y_pred
    all_labels = sorted(set(y_true) | set(y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=[idx_to_chord[i] for i in all_labels],
        zero_division=0
    ))

evaluate_with_metrics(model, x_test, y_test)

# -----------------------------
# Harmonize New Melody
# -----------------------------
def harmonize(melody_notes):
    model.eval()
    input_idxs = [note_to_idx.get(n, note_to_idx["UNK"]) for n in melody_notes]
    x = torch.tensor(input_idxs).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
        pred_idxs = pred.argmax(-1).squeeze().tolist()
    return [idx_to_chord[i] for i in pred_idxs]

# -----------------------------
# Save Harmonized MIDI
# -----------------------------
def save_to_midi(melody, chords, filename="harmonized.mid"):
    s = stream.Stream()
    for m, c in zip(melody, chords):
        n = note.Note(m) if m != "Rest" else note.Rest()
        n.quarterLength = 1.0
        s.append(n)
    s.write("midi", fp=filename)

# -----------------------------
# Example Run
# -----------------------------
input_melody = all_melodies[0][:30]
predicted_chords = harmonize(input_melody)
print("Melody:", input_melody)
print("Predicted Chords:", predicted_chords)
save_to_midi(input_melody, predicted_chords)
