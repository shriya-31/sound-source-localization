import numpy as np
import gc
import nengo
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class DecoderTrainingData:
    X: np.ndarray #(n_samples, n_channels)
    angles_deg: np.ndarray #(n_samples,) - angles

class ICDecoderInput:
    def __init__(self, n_channels):
        self.data = np.zeros(n_channels)
    def __call__(self, t):
        return self.data

class ICCircularDecoder:
    def __init__(self, config, n_neurons=1000, radius=None, reg=0.01, settle_s=0.05):
        self.config = config
        self.n_neurons = n_neurons
        self.radius = radius
        self.reg = reg
        self.settle_s = settle_s
        self.model = None
        self.probe = None
        self.input_ctrl = None
        self.sim = None

    def fit(self, X_train, y_train_deg):
        n_channels = X_train.shape[1]
        self.input_ctrl = ICDecoderInput(n_channels)

        theta = np.deg2rad(y_train_deg)
        y_train_cos_sin = np.column_stack([np.cos(theta), np.sin(theta)]) 

        if self.radius is None:
            self.radius = np.max(np.linalg.norm(X_train, axis=1)) * 1.05

        with nengo.Network(seed=0) as model:
            ild_input = nengo.Node(self.input_ctrl, size_out=n_channels)

            ic = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=n_channels,
                radius=self.radius,
            )
            nengo.Connection(ild_input, ic, synapse=self.config.tau_excit)

            az_node = nengo.Node(size_in=2)
            nengo.Connection(
                ic, az_node,
                eval_points=X_train,
                function=y_train_cos_sin,      
                solver=nengo.solvers.LstsqL2(reg=self.reg),
                scale_eval_points=False,
            )
            self.probe = nengo.Probe(az_node, synapse=self.config.smooth_ms / 1000)

        self.model = model
        self.sim = nengo.Simulator(self.model, dt=self.config.dt_sim, progress_bar=False)
        return self

    def predict_one(self, ild_vector: np.ndarray):
        if self.sim is None:
            raise RuntimeError("Call fit().")
        self.input_ctrl.data = ild_vector
        self.sim.run(self.settle_s)
        pred_cos_sin = self.sim.data[self.probe][-1, :]
        self.sim.reset()
        angle = np.rad2deg(np.arctan2(pred_cos_sin[1], pred_cos_sin[0]))
        angle = angle % 360
        return float(angle)

    def predict(self, X: np.ndarray):
        return np.array([self.predict_one(x) for x in X])

    def close(self):
        if self.sim is not None:
            self.sim.close()


class ICDecoderEvaluator:

    @staticmethod
    def calculate_circular_errors(y_true, y_pred):
        y_true = np.asarray(y_true) % 360
        y_pred = np.asarray(y_pred) % 360
        return ((y_pred - y_true + 180) % 360) - 180
    
    @staticmethod
    def evaluate(decoder: ICCircularDecoder, X_test, y_test_deg):
        preds = decoder.predict(X_test)
        errors = ICDecoderEvaluator.calculate_circular_errors(y_test_deg, preds)
        mae = np.mean(np.abs(errors))
        return preds, mae

    @staticmethod
    def plot(actual_deg, predicted_deg, mae, title="Decoder"):
        plt.figure(figsize=(6, 6))
        plt.scatter(actual_deg, predicted_deg, s=80)
        min_val, max_val = min(actual_deg), max(actual_deg)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Real azimuth (°)')
        plt.ylabel('Predicted azimuth (°)')
        plt.title(f'{title} - MAE = {mae:.1f}°')
        plt.grid(True, alpha=0.3)
        plt.show()
