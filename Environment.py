import numpy as np
import random
import tensorflow as tf
class SequenceEnvironment:
    def __init__(self, seq_length, vocab_size, batch_size, surrogate):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.surrogate = surrogate

    def reset_targets(self):
        self.gc_target = np.random.uniform(0.3, 0.7)
        self.motif_target_index = random.randint(4, self.seq_length - 4)
        self.motif_target_base = random.randint(0, 3)

    def decode_sequence(self, sequences):
        idx_to_nt = ['A', 'C', 'G', 'U']
        return ["".join(idx_to_nt[i] for i in seq) for seq in sequences]

    def surrogate_reward(self, sequences):
        raw_scores = tf.squeeze(self.surrogate(sequences), axis=1)
        clipped = tf.clip_by_value(raw_scores, -50.0, 0.0)  # realistic RNAfold energies
        return clipped

    def gc_content_reward(self, sequences):
        is_gc = np.logical_or(sequences == 1, sequences == 2)
        gc_ratio = np.mean(is_gc, axis=1)
        deviation = (gc_ratio - self.gc_target) ** 2
        return tf.constant(1.0 - 5.0 * deviation, dtype=tf.float32)

    def motif_reward(self, sequences):
        match = sequences[:, self.motif_target_index] == self.motif_target_base
        penalty = -1.0 * np.max((sequences != self.motif_target_base).astype(np.float32), axis=1)
        return tf.constant(match.astype(np.float32) * 2.0 + penalty, dtype=tf.float32)

    def structure_entropy_reward(self, probs):
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        mean_entropy = np.mean(entropy, axis=1)
        return tf.constant(1.0 - np.clip(mean_entropy, 0.0, 1.0), dtype=tf.float32)

    def repeat_penalty(self, sequences):
        penalties = []
        for seq in sequences:
            counts = [np.sum(seq == nt) for nt in range(self.vocab_size)]
            avg_allowed = self.seq_length // self.vocab_size + 1
            excesses = [max(0, c - avg_allowed) for c in counts]
            penalty = -1.0 * sum(excesses)
            penalties.append(penalty)
        return tf.constant(penalties, dtype=tf.float32)

    def compute_rewards(self, sequences, probs):
        self.reset_targets()
        surrogate_score = self.surrogate_reward(sequences)
        gc_score = self.gc_content_reward(sequences)
        motif_score = self.motif_reward(sequences)
        entropy_score = self.structure_entropy_reward(probs)
        repeat_pen = self.repeat_penalty(sequences)
        return surrogate_score + gc_score + motif_score + 1.0 * entropy_score + repeat_pen
