from memoryattend import MemoFormer
from memoryattend import FoldingSurrogate
from memoryattend import ImprovedSurrogate
import tensorflow as tf
from Environment import SequenceEnvironment
from memoryattend import TransformerBaseline
import os
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import tempfile
import subprocess
import matplotlib.pyplot as plt
import pandas as pd



def train_memoformer(USE_MEMORY,USE_IMPROVED_SURROGATE,MEMORY_TYPE):
    seq_length = 30
    batch_size = 32
    vocab_size = 4
    reward_history = []
    if USE_IMPROVED_SURROGATE:
        surrogate = ImprovedSurrogate(vocab_size=vocab_size, seq_length=seq_length)
        surrogate_tag = "improved"
    else:
        surrogate = FoldingSurrogate(vocab_size=vocab_size, seq_length=seq_length)
        surrogate_tag = "folding"

    if USE_MEMORY:
        model = MemoFormer(d_model=64, num_heads=4, ff_dim=128, vocab_size=vocab_size, memory_dim=64,memory_type= MEMORY_TYPE)
        tag = f"memoformer_{surrogate_tag}_{MEMORY_TYPE}_"
    else:
        model = TransformerBaseline(d_model=64, num_heads=4, ff_dim=128, vocab_size=vocab_size)
        tag = f"baseline_{surrogate_tag}"

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    env = SequenceEnvironment(seq_length, vocab_size, batch_size, surrogate)

    for epoch in range(10000):
        input_dummy = tf.zeros((batch_size, seq_length, vocab_size))
        with tf.GradientTape() as tape:
            logits = model(input_dummy)
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            sampled = tf.random.categorical(tf.reshape(logits, [-1, vocab_size]), 1)
            sampled = tf.reshape(sampled, [batch_size, seq_length])
            rewards = env.compute_rewards(sampled.numpy(), probs)
            reward_history.append(tf.reduce_mean(rewards).numpy())
            baseline = tf.reduce_mean(rewards)
            advantage = rewards - baseline

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_samples = tf.one_hot(sampled, depth=vocab_size)
            selected_log_probs = tf.reduce_sum(log_probs * one_hot_samples, axis=-1)

            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
            entropy_bonus = tf.reduce_mean(entropy)

            loss = -tf.reduce_mean(advantage * tf.reduce_sum(selected_log_probs, axis=1)) - 0.01 * entropy_bonus

        grads = tape.gradient(loss, model.trainable_variables + surrogate.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables + surrogate.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.numpy():.4f} | Avg Reward: {tf.reduce_mean(rewards).numpy():.4f}")

        if epoch % 1000 == 0:
            decoded = env.decode_sequence(sampled.numpy())
            with open(f"results/{tag}_epoch_{epoch}.txt", "w") as f:
                for s in decoded:
                    f.write(s + "\n")
            plot_sequence_logo(probs, f"results/sequence_logo_{tag}_{epoch}.png")
    np.save(f"results/reward_history_{tag}.npy", reward_history)
    return model, surrogate

def evaluate_surrogate_accuracy(surrogate_model, sequences, model_tag,surrogate_tag,memory_config, batch_size=32):
    idx_to_nt = ['A', 'C', 'G', 'U']
    sequence_strs = ["".join([idx_to_nt[i] for i in seq]) for seq in sequences]
    true_energies = []
    for seq in sequence_strs:
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp:
            temp.write(seq)
            path = temp.name
        try:
            result = subprocess.run(["RNAfold", path], capture_output=True, text=True)
            energy_line = result.stdout.strip().split("\n")[-1]
            energy = float(energy_line.split('(')[-1].split(')')[0])
            true_energies.append(-energy)
        except:
            true_energies.append(0.0)
        finally:
            os.unlink(path)

    sequences_tensor = tf.convert_to_tensor(sequences, dtype=tf.int32)
    predicted = tf.squeeze(surrogate_model(sequences_tensor)).numpy()

    df = pd.DataFrame({"True Energy": true_energies, "Surrogate Prediction": predicted})
    df.plot.scatter(x="True Energy", y="Surrogate Prediction", title="Surrogate vs ViennaRNA")
    plt.tight_layout()
    plt.savefig(f"results/{model_tag}_{surrogate_tag}_{memory_config}_surrogate_vs_rnafold.png")
    plt.close()

def plot_sequence_logo(probs, filename):
    avg_probs = np.mean(probs, axis=0)  # shape: [seq_length, vocab_size]
    fig, ax = plt.subplots(figsize=(10, 3))
    positions = np.arange(avg_probs.shape[0])
    for i, base in enumerate(['A', 'C', 'G', 'U']):
        ax.bar(positions, avg_probs[:, i], bottom=np.sum(avg_probs[:, :i], axis=1), label=base)
    ax.set_xticks(positions)
    ax.set_xticklabels(positions + 1)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Base Frequency")
    ax.legend(loc="upper right")
    ax.set_title("Sequence Logo from Transformer Output")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_sequence_space(sequences, surrogate_model,model_tag,surrogate_tag, memory_config):
    x = tf.convert_to_tensor(sequences)
    embeddings = surrogate_model.embedding(x).numpy()  # [batch, seq_len, dim]
    pooled = embeddings.mean(axis=1)  # average over sequence positions
    energies = tf.squeeze(surrogate_model(x)).numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(pooled)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=energies, cmap="viridis", s=40)
    plt.colorbar(sc, label="Surrogate Energy")
    plt.title("Sequence Embedding Space by Surrogate Energy")
    plt.tight_layout()
    plt.savefig(f"results/{model_tag}_{surrogate_tag}_{memory_config}_sequence_space_energy.png")
    plt.close()

def plot_comparative_rewards(results_dir="results"):
    import glob
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(10, 6))
    for file in sorted(glob.glob(f"{results_dir}/reward_history_*.npy")):
        rewards = np.load(file)
        label = os.path.basename(file).replace("reward_history_", "").replace(".npy", "")
        plt.plot(rewards, label=label)

    plt.title("Average Reward Across Training")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/reward_comparison.png")
    plt.close()
    print("Saved reward comparison to results/reward_comparison.png")




import imageio

def generate_training_gif(image_dir="results", output_gif="results/training_progress.gif"):
    import glob
    files = sorted(glob.glob(os.path.join(image_dir, "sequence_logo_memoformer_improved_*.png")))
    images = [imageio.imread(f) for f in files if os.path.exists(f)]
    if images:
        imageio.mimsave(output_gif, images, fps=2)
    print("GIF saved to", output_gif)

# End of update

#plot_comparative_rewards()
#generate_training_gif()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    configs = [
        #(True, False, 'static'),  # MemoFormer + Improved
        #(False, True, 'static'),  # Baseline + Improved
        #(False, False, 'static'),  # Baseline + Folding
        (True, True, 'static'),
        (True, True, 'lstm'),
        (True, True, 'minimal_gru')
    ]


    for use_memory, use_improved, memory_config in configs:
        print(
            f"\n=== Training: {'MemoFormer' if use_memory else 'Baseline'} + {'Improved' if use_improved else 'Folding'} + {memory_config} ===")
        model, surrogate = train_memoformer(USE_MEMORY=use_memory, USE_IMPROVED_SURROGATE=use_improved, MEMORY_TYPE = memory_config)

        # Evaluate
        seq_length = 30
        batch_size = 32
        vocab_size = 4
        dummy_env = SequenceEnvironment(seq_length, vocab_size, batch_size, surrogate)
        dummy_env.reset_targets()
        input_dummy = tf.zeros((batch_size, seq_length, vocab_size))
        logits = model(input_dummy)
        sampled = tf.random.categorical(tf.reshape(logits, [-1, vocab_size]), 1)
        sampled = tf.reshape(sampled, [batch_size, seq_length]).numpy()
        model_tag = "memoformer" if use_memory else "baseline"
        surrogate_tag = "improved" if use_improved else "folding"
        evaluate_surrogate_accuracy(surrogate,sampled,model_tag,surrogate_tag, memory_config, batch_size)
        visualize_sequence_space(sampled, surrogate, model_tag, surrogate_tag,memory_config)
        probs = tf.nn.softmax(logits, axis=-1).numpy()

        plot_sequence_logo(probs, f"results/sequence_logo_final_{model_tag}_{surrogate_tag}_{memory_config}.png")
        model.save_weights(f"results/{model_tag}_{surrogate_tag}_{memory_config}_weights.h5")
        surrogate.save_weights(f"results/{surrogate_tag}_surrogate_weights.h5")

    plot_comparative_rewards()
    generate_training_gif()