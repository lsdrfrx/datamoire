from ._utils import optimizers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from dataclasses import dataclass, asdict, field
import pandas as pd
from typing import Any

import plotly.graph_objs as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class Parameters:
    BATCH_SIZE: int = field(default=None)
    HIDDEN_DIM: list = field(default=None)
    DROPOUT: float = field(default=None)
    LSTM_LAYERS: int = field(default=None)
    OPTIMIZER: Any = field(default=None)
    LEARNING_RATE: float = field(default=None)

    def __repr__(self):
        return '\n'.join([f"{key}:\t{value}" for key, value in asdict(self).items()])


class ParameterTuner:
    def __init__(self, model, params_list, device=torch.device("cpu")):
        self.summary = pd.DataFrame(columns=["candidates", "best loss", "best metric"])

        self.model = model
        self.params_list = params_list
        self.N_EPOCHS = 200
        self.data = None
        self.device = device

        self.criterion = None


    def set_criterion(self, criterion):
        self.criterion = criterion
    

    def set_epochs(self, N_EPOCHS):
        self.N_EPOCHS = N_EPOCHS


    def set_k_fold(X, y):
        pass


    def set_leave_one_out(X, y):
        pass


    def set_hold_out(X, y):
        pass


    def summary(self):
        pass


    def fit(self, data):
        print(f"Train epochs count: {self.N_EPOCHS}\n")
        for i, params in enumerate(self.params_list):
            print(f"Training canditate #{i} with:")
            print(params)

            model = self.model(
                dropout=params.DROPOUT,
                n_hidden=params.N_HIDDEN,
                lstm_layers=params.LSTM_LAYERS,
            ).to(self.device)
                
            optimizer = params.OPTIMIZER(model.parameters(), lr=params.LEARNING_RATE)
            criterion = self.criterion()

            train_losses = []
            test_losses = []

            train_acc = []
            test_acc = []

            accuracy = Accuracy(task="multiclass", num_classes=2).to(device)

            for epoch in trange(
                self.N_EPOCHS, desc=f"Candidate #{i} training"
            ):
                total_acc_train = 0
                total_loss_train = 0

                for features, expected in train:
                    features = features.to(device)
                    expected = expected.to(device)

                    optimizer.zero_grad()

                    model.hidden_state = (
                        torch.zeros(model.lstm_layers, model.n_hidden[0]).to(device),
                        torch.zeros(model.lstm_layers, model.n_hidden[0]).to(device),
                    )

                    output = model(features)

                    loss = criterion(output, expected)

                    total_loss_train += loss.item()

                    pred = torch.argmax(output, dim=1)
                    label = torch.argmax(expected, dim=1)
                    total_acc_train += accuracy(pred, label)

                    loss.backward()
                    optimizer.step()

                acc = total_acc_train / len(train)
                loss = total_loss_train / len(train)

                train_losses.append(loss)
                train_acc.append(float(acc.cpu()))

                total_acc_test = 0
                total_loss_test = 0

                with torch.no_grad():
                    for features, expected in test:
                        output = model(features)
                        pred = torch.argmax(output, dim=1)
                        label = torch.argmax(expected, dim=1)

                        loss = criterion(output, expected)

                        total_loss_test += loss.item()
                        total_acc_test += accuracy(pred, label)

                acc = total_acc_test / len(test)
                loss = total_loss_test / len(test)

                test_losses.append(loss)
                test_acc.append(float(acc.cpu()))

            plot, mean_acc = self._make_plot(
                train_losses, test_losses, train_acc, test_acc, f"candidate #{i}"
            )
            self.summary_dict[f"candidate #{i}"] = {
                "plot": plot,
                "mean_accuracy": mean_acc,
                "parameters": params,
            }

            print(
                f"Candidate #{i} finished training with mean test accuracy: {mean_acc}\n"
            )


    def _make_plot(self, train_losses, test_losses, train_acc, test_acc, title):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.02, shared_xaxes=True)

        test_acc[:5] = [0 for _ in range(5)]

        maximums = sorted(test_acc, reverse=True)[:3]

        # Потери на обучающей и тестовой выборке
        fig.add_trace(
            go.Scatter(
                x=list(range(1, N_EPOCHS + 1)),
                y=train_losses,
                mode="lines",
                name="train loss",
            ),
            row=1,
            col=1,
        ).add_trace(
            go.Scatter(
                x=list(range(1, N_EPOCHS + 1)),
                y=test_losses,
                mode="lines",
                name="test loss",
            ),
            row=1,
            col=1,
        )

        # Точность на обучающей и тестовой выборке
        fig.add_trace(
            go.Scatter(
                x=list(range(1, N_EPOCHS + 1)),
                y=train_acc,
                mode="lines",
                name="train accuracy",
            ),
            row=2,
            col=1,
        ).add_trace(
            go.Scatter(
                x=list(range(1, N_EPOCHS + 1)),
                y=test_acc,
                mode="lines",
                name="test accuracy",
            ),
            row=2,
            col=1,
        )

        for maximum in maximums:
            max_index = test_acc.index(maximum)
            loss = test_losses[max_index]

            # Маркеры лучших значений
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, N_EPOCHS + 1)),
                    y=[None if i != maximum else maximum for i in test_acc],
                    mode="markers",
                    marker_color="red",
                    name="best test accuracy",
                ),
                row=2,
                col=1,
            ).add_trace(
                go.Scatter(
                    x=list(range(1, N_EPOCHS + 1)),
                    y=[None if i != maximum else loss for i in test_acc],
                    mode="markers",
                    marker_color="red",
                    name="best epoch test loss",
                ),
                row=1,
                col=1,
            )

            # Пунктирные вертикальные линии
            fig.add_vline(row=1, col=1, x=max_index + 1, line_dash="dash", line_width=1)

            fig.add_vline(
                row=2,
                col=1,
                x=max_index + 1,
                line_dash="dash",
                line_width=1,
                label=dict(
                    text=f"Epoch: {max_index + 1}",
                    textposition="start",
                ),
            )

        fig.update_layout(title_text=f"<b>{title}</b>", title_x=0.5, height=600)
        fig.update_xaxes(title_text="Эпохи", row=2, col=1)
        fig.update_yaxes(title_text="Потери", row=1, col=1)
        fig.update_yaxes(title_text="Точность", row=2, col=1)

        return fig, sum(maximums) / 3

    
