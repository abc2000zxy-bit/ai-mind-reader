import random
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window

Window.clearcolor = (0.96, 0.97, 0.98, 1)

MOVES = ["ROCK", "PAPER", "SCISSORS"]
BEATS = {0: 1, 1: 2, 2: 0}

class MiniNeuralNet:
    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(6, 8) * 0.5
        self.b1 = np.zeros((1, 8))
        self.W2 = np.random.randn(8, 3) * 0.5
        self.b2 = np.zeros((1, 3))

    def _encode_pair(self, m1, m2):
        vec = np.zeros((1, 6))
        vec[0, m1] = 1
        vec[0, 3 + m2] = 1
        return vec

    def train(self, X_pairs, y_moves, epochs=120, lr=0.15):
        X = np.vstack([self._encode_pair(a, b) for a, b in X_pairs])
        Y = np.zeros((len(y_moves), 3))
        for i, m in enumerate(y_moves):
            Y[i, m] = 1

        for _ in range(epochs):
            z1 = X @ self.W1 + self.b1
            a1 = np.tanh(z1)
            z2 = a1 @ self.W2 + self.b2
            exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            dz2 = (probs - Y) / len(X)
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - a1 ** 2)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

    def predict(self, m1, m2):
        x = self._encode_pair(m1, m2)
        a1 = np.tanh(x @ self.W1 + self.b1)
        z2 = a1 @ self.W2 + self.b2
        return int(np.argmax(z2))


class MindReaderApp(App):
    def build(self):
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
            text="You: 0   |   AI: 0   |   Draws: 0",
            font_size="18sp", bold=True, color=(0.15, 0.2, 0.3, 1), size_hint=(1, 0.1)
        )
        layout.add_widget(self.score_label)

        self.status_label = Label(
            text="Tap a move below to start playing!",
            font_size="20sp", bold=True, halign="center",
            color=(0.2, 0.2, 0.2, 1), size_hint=(1, 0.3)
        )
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
            text="Status: Warm-up (Play 5 rounds to activate AI)",
            font_size="14sp", color=(0.1, 0.3, 0.7, 1), size_hint=(1, 0.12)
        )
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

        self.status_label.text = f"You: {MOVES[player_move]}   |   AI: {MOVES[ai_move]}\n\n{outcome}"
        self.status_label.color = color
        self.score_label.text = f"You: {self.scores['Player']}   |   AI: {self.scores['AI']}   |   Draws: {self.scores['Draw']}"
        self.mode_label.text = f"Status: {ai_mode} | Rounds Learned: {len(self.X_train)}"


if __name__ == "__main__":
    MindReaderApp().run()
