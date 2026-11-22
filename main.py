import math
import random
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window

Window.clearcolor = (0.96, 0.97, 0.98, 1)

MOVES = ["ROCK", "PAPER", "SCISSORS"]
BEATS = {0: 1, 1: 2, 2: 0}


class MiniNeuralNet:
    """6 inputs (one-hot of last 2 moves) -> 8 hidden (tanh) -> 3 outputs (softmax). Pure Python."""

    def __init__(self):
        r = random.Random(42)
        self.W1 = [[r.gauss(0, 0.5) for _ in range(8)] for _ in range(6)]
        self.b1 = [0.0] * 8
        self.W2 = [[r.gauss(0, 0.5) for _ in range(3)] for _ in range(8)]
        self.b2 = [0.0] * 3

    def _encode_pair(self, m1, m2):
        v = [0.0] * 6
        v[m1] = 1.0
        v[3 + m2] = 1.0
        return v

    def _forward(self, x):
        a1 = [math.tanh(sum(x[i] * self.W1[i][j] for i in range(6)) + self.b1[j])
              for j in range(8)]
        z2 = [sum(a1[j] * self.W2[j][k] for j in range(8)) + self.b2[k]
              for k in range(3)]
        mx = max(z2)
        e = [math.exp(z - mx) for z in z2]
        s = sum(e)
        return a1, [v / s for v in e]

    def train(self, X_pairs, y_moves, epochs=120, lr=0.15):
        X = [self._encode_pair(a, b) for a, b in X_pairs]
        n = len(X)
        for _ in range(epochs):
            dW1 = [[0.0] * 8 for _ in range(6)]
            db1 = [0.0] * 8
            dW2 = [[0.0] * 3 for _ in range(8)]
            db2 = [0.0] * 3
            for x, y in zip(X, y_moves):
                a1, p = self._forward(x)
                dz2 = [(p[k] - (1.0 if k == y else 0.0)) / n for k in range(3)]
                for k in range(3):
                    db2[k] += dz2[k]
                for j in range(8):
                    for k in range(3):
                        dW2[j][k] += a1[j] * dz2[k]
                    da = sum(dz2[k] * self.W2[j][k] for k in range(3))
                    dz1 = da * (1 - a1[j] ** 2)
                    db1[j] += dz1
                    for i in range(6):
                        dW1[i][j] += x[i] * dz1
            for i in range(6):
                for j in range(8):
                    self.W1[i][j] -= lr * dW1[i][j]
            for j in range(8):
                self.b1[j] -= lr * db1[j]
                for k in range(3):
                    self.W2[j][k] -= lr * dW2[j][k]
            for k in range(3):
                self.b2[k] -= lr * db2[k]

    def predict(self, m1, m2):
        _, p = self._forward(self._encode_pair(m1, m2))
        return p.index(max(p))


class MindReaderApp(App):
    def build(self):
        self.title = "AI Mind Reader"
        self.history = []
        self.X_train = []
        self.y_train = []
        self.scores = {"Player": 0, "AI": 0, "Draw": 0}
        self.brain = MiniNeuralNet()
        self.is_trained = False

        layout = BoxLayout(orientation="vertical", padding=25, spacing=15)

        layout.add_widget(Label(
            text="AI Mind Reader", font_size="28sp", bold=True,
            color=(0.1, 0.15, 0.25, 1), size_hint=(1, 0.12)
        ))

        self.score_label = Label(
            text="You: 0 | AI: 0 | Draws: 0",
            font_size="18sp", bold=True, color=(0.15, 0.2, 0.3, 1),
            size_hint=(1, 0.1)
        )
        layout.add_widget(self.score_label)

        self.status_label = Label(
            text="Tap a move below to start playing!",
            font_size="20sp", bold=True, halign="center", valign="middle",
            color=(0.2, 0.2, 0.2, 1), size_hint=(1, 0.3)
        )
        self.status_label.bind(
            size=lambda inst, val: setattr(inst, "text_size", val))
        layout.add_widget(self.status_label)

        colors = [(0.98, 0.8, 0.2, 1), (0.4, 0.7, 0.95, 1), (0.45, 0.8, 0.5, 1)]
        for idx, name in enumerate(MOVES):
            btn = Button(
                text=name, font_size="20sp", bold=True,
                background_normal="", background_color=colors[idx],
                color=(0.1, 0.1, 0.1, 1), size_hint=(1, 0.12)
            )
            btn.bind(on_press=lambda instance, m=idx: self.play_round(m))
            layout.add_widget(btn)

        self.mode_label = Label(
            text="Status: Warm-up (Play 7 rounds to activate AI)",
            font_size="14sp", halign="center", valign="middle",
            color=(0.1, 0.3, 0.7, 1), size_hint=(1, 0.12)
        )
        self.mode_label.bind(
            size=lambda inst, val: setattr(inst, "text_size", val))
        layout.add_widget(self.mode_label)

        return layout

    def play_round(self, player_move):
        if self.is_trained and len(self.history) >= 2:
            pred = self.brain.predict(self.history[-2], self.history[-1])
            ai_move = BEATS[pred]
            ai_mode = "Neural Network Active [AI]"
        else:
            ai_move = random.randint(0, 2)
            ai_mode = "Random (Gathering Data...)"

        if player_move == ai_move:
            outcome, color = "It's a Draw!", (0.35, 0.35, 0.35, 1)
            self.scores["Draw"] += 1
        elif BEATS[ai_move] == player_move:
            outcome, color = "You Win!", (0.1, 0.6, 0.2, 1)
            self.scores["Player"] += 1
        else:
            outcome, color = "AI Predicted You & Won!", (0.85, 0.15, 0.15, 1)
            self.scores["AI"] += 1

        if len(self.history) >= 2:
            self.X_train.append((self.history[-2], self.history[-1]))
            self.y_train.append(player_move)
            if len(self.X_train) >= 5:
                self.brain.train(self.X_train, self.y_train)
                self.is_trained = True

        self.history.append(player_move)

        self.status_label.text = (
            "You: %s | AI: %s\n\n%s" % (MOVES[player_move], MOVES[ai_move], outcome))
        self.status_label.color = color
        self.score_label.text = "You: %d | AI: %d | Draws: %d" % (
            self.scores["Player"], self.scores["AI"], self.scores["Draw"])
        self.mode_label.text = "Status: %s | Rounds Learned: %d" % (
            ai_mode, len(self.X_train))


if __name__ == "__main__":
    MindReaderApp().run()
