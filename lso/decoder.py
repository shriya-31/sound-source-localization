import numpy as np
import gc
import nengo
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class DecoderTrainingData:
    X: np.ndarray #(n_samples, n_channels)
    y: np.ndarray #(n_samples,) - normed angles [-1, 1]
    angles_deg: np.ndarray #(n_samples,) - angles

class DatasetBuilder:
    def __init__(self, config, brainstem, spatializer, stimulus_gen):
        self.config = config
        self.brainstem = brainstem
        self.spatializer = spatializer
        self.stimulus_gen = stimulus_gen

    def build(self, seeds, angles):
        X, y, angs = [], [], []
        for seed in seeds:
            mono = self.stimulus_gen.generate_noise(seed=seed)
            for angle in angles:
                left, right = self.spatializer.spatialize(mono, angle)
                res = self.brainstem.run(left, right)
                X.append(res.ild)
                y.append(angle / 90.0)
                angs.append(angle)
                #gc.collect()
        return DecoderTrainingData(np.array(X), np.array(y), np.array(angs))

class ICDecoderInput:
    def __init__(self, n_channels):
        self.data = np.zeros(n_channels)
    def __call__(self, t):
        return self.data

class ICDecoder:
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

    def fit(self, X_train, y_train):
        n_channels = X_train.shape[1]
        self.input_ctrl = ICDecoderInput(n_channels)

        y_train_2d = np.asarray(y_train).reshape(-1, 1)

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

            az_node = nengo.Node(size_in=1)
            nengo.Connection(
                ic, az_node,
                eval_points=X_train,
                function=y_train_2d,      
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
        pred_norm = self.sim.data[self.probe][-1, 0]
        self.sim.reset()
        return pred_norm * 90.0

    def predict(self, X: np.ndarray):
        return np.array([self.predict_one(x) for x in X])

    def close(self):
        if self.sim is not None:
            self.sim.close()


class ICDecoderEvaluator:
    @staticmethod
    def evaluate(decoder: ICDecoder, X_test, y_test_deg):
        preds = decoder.predict(X_test)
        mae = np.mean(np.abs(preds - y_test_deg))
        return preds, mae

    @staticmethod
    def plot(actual_deg, predicted_deg, mae, title="Decoder"):
        plt.figure(figsize=(6, 6))
        plt.scatter(actual_deg, predicted_deg, s=80)
        plt.plot([-90, 90], [-90, 90], 'r--')
        plt.xlabel('Real azimuth (°)')
        plt.ylabel('Predicted azimuth (°)')
        plt.title(f'{title} - MAE = {mae:.1f}°')
        plt.grid(True, alpha=0.3)
        plt.show()