import math
import json
import itertools
from Bio import SeqIO

def read_model(path):
    """
    Input: path of a model file.
    Output: an HMM.
    Example: read_model("weather_model.json").states → ["H", "L"], .alphabet → ["s", "r"].
    """
    with open(path) as f:
        model = json.load(f)
    return HMM(model["states"], model["alphabet"], model["pi"], model["a"], model["e"])


def log2(p):
    """
    Input: p, a probability (a float ≥ 0).
    Output: float, math.log2(p), or -math.inf when p is 0. 
    """
    p = float(p)
    if p==0:
        return -math.inf
    else:
        return math.log2(p)

def log_sum(a, b):
    """
    Input: a, b, two log2 values (possibly -math.inf).
    Output: float, log2(2^a + 2^b), computed without leaving log space: max + log2(1 + 2^(min − max)).
    Examples: log_sum(3, 3) → 4.0; log_sum(0, -math.inf) → 0.0; log_sum(-math.inf, -math.inf) → -inf.
    """
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    min_val = min(a, b)
    max_val = max(a, b)
    return max_val + log2(1 + 2**(min_val - max_val))



class HMM:
    def __init__(self, states, alphabet, pi, a, e):
        self.states = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e = e
        self.log_pi = {s: log2(p) for s, p in pi.items()}
        self.log_a = {s: {t: log2(p) for t, p in row.items()} for s, row in a.items()}
        self.log_e = {s: {c: log2(p) for c, p in row.items()} for s, row in e.items()}


    def joint_probability(self,x, z):
        """
        Input: a sequence x and a path z of the same length.
        Output: float, P(x, z) = π(z₀)·e(z₀, x₀) · a(z₀→z₁)·e(z₁, x₁) · … · a(zₙ₋₂→zₙ₋₁)·e(zₙ₋₁, xₙ₋₁).
        Example (weather): joint_probability("sr", "HL") → 0.054 (0.5 · 0.9 · 0.2 · 0.6).
        """
        probability = self.pi[z[0]] * self.e[z[0]][x[0]]
        for i in range(1, len(x)):
            probability *= self.a[z[i - 1]][z[i]] * self.e[z[i]][x[i]]
        return probability

    def log_joint(self, x, z):
        """
        Input: as joint_probability.
        Output: float, log2 P(x, z), computed as a sum of the log2 of every factor, taken from self.log_pi, self.log_a and self.log_e (not as the log2 of the product).
        Example (weather): log_joint("sr", "HL") → about −4.21.
        """
        log_probability = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]
        for i in range(1, len(x)):
            log_probability += self.log_a[z[i - 1]][z[i]] + self.log_e[z[i]][x[i]]
        return log_probability

    def brute_force(self, x):
        """
        Input: a sequence x (short!).
        Output: tuple[str, float, float]: the most likely path, its probability, and the sum of joint_probability(x, z) over all paths z.
        Example (weather): brute_force("ssrssrrrss") → ("HHHHHLLLHH", 5.53e-05, 0.000822) (rounded; the lecture's numbers).
        """
        all_paths = list(itertools.product(self.states, repeat=len(x)))
        max_path = ""
        max_prob = 0
        total_prob = 0
        for path in all_paths:
            prob = self.joint_probability(x, path)
            total_prob += prob
            if prob > max_prob:
                max_prob = prob
                max_path = "".join(path)
        return max_path, max_prob, total_prob

    def viterbi(self, x):
        """
        Input: a sequence x.
        Output: tuple[str, float], the most likely hidden path and its log2 probability.
        Example (weather): viterbi("ssrssrrrss") → ("HHHHHLLLHH", -14.14...).
        """
        v = [{} for _ in x]
        back = [{} for _ in x]
        for k in self.states:
            v[0][k] = self.log_pi[k] + self.log_e[k][x[0]]
        for i in range(1, len(x)):
            for k in self.states:
                best_prev = self.states[0]
                best = v[i - 1][best_prev] + self.log_a[best_prev][k]
                for j in self.states[1:]:
                    score = v[i - 1][j] + self.log_a[j][k]
                    if score > best:
                        best = score
                        best_prev = j
                v[i][k] = best + self.log_e[k][x[i]]
                back[i][k] = best_prev
        last = self.states[0]
        for k in self.states[1:]:
            if v[-1][k] > v[-1][last]:
                last = k
        path = [last]
        for i in range(len(x) - 1, 0, -1):
            path.append(back[i][path[-1]])
        return "".join(reversed(path)), v[-1][last]

    def vitebri_plain(self, x):
        """
        Input: a sequence x.
        Output: tuple[str, float], the most likely hidden path and its probability.
        """
        v = [{} for _ in x]
        back = [{} for _ in x]
        for k in self.states:
            v[0][k] = self.pi[k] * self.e[k][x[0]]
        for i in range(1, len(x)):
            for k in self.states:
                best_prev = self.states[0]
                best = v[i - 1][best_prev] * self.a[best_prev][k]
                for j in self.states[1:]:
                    score = v[i - 1][j] * self.a[j][k]
                    if score > best:
                        best = score
                        best_prev = j
                v[i][k] = best * self.e[k][x[i]]
                back[i][k] = best_prev
        last = self.states[0]
        for k in self.states[1:]:
            if v[-1][k] > v[-1][last]:
                last = k
        path = [last]
        for i in range(len(x) - 1, 0, -1):
            path.append(back[i][path[-1]])
        return "".join(reversed(path)), v[-1][last]

    def forward(self, x):
        """
        Input: a sequence x.
        Output: tuple[list[dict], float]: the table f, f[i][k] = log2 P(x₀ … xᵢ, zᵢ = k), and log2 P(x).
        Example (weather): the log2 P(x) of "ssrssrrrss" is about −10.25 (P(x) = 0.000822).
        """
        f = [{} for _ in x]
        for k in self.states:
            f[0][k] = self.log_pi[k] + self.log_e[k][x[0]]
        for i in range(1, len(x)):
            for k in self.states:
                total = -math.inf
                for j in self.states:
                    total = log_sum(total, f[i - 1][j] + self.log_a[j][k])
                f[i][k] = total + self.log_e[k][x[i]]
        log_prob_x = -math.inf
        for k in self.states:
            log_prob_x = log_sum(log_prob_x, f[-1][k])
        return f, log_prob_x

    def backward(self, x):
        """
        Input: a sequence x.
        Output: tuple[list[dict], float]: the table b, b[i][k] = log2 P(xᵢ₊₁ … xₙ₋₁ | zᵢ = k), and log2 P(x).
        Example (weather): the log2 P(x) of "ssrssrrrss" is about −10.25 (P(x) = 0.000822).
        """
        b = [{} for _ in x]
        for k in self.states:
            b[-1][k] = 0
        for i in range(len(x) - 2, -1, -1):
            for k in self.states:
                total = -math.inf
                for j in self.states:
                    total = log_sum(total, self.log_a[k][j] + self.log_e[j][x[i + 1]] + b[i + 1][j])
                b[i][k] = total
        log_prob_x = -math.inf
        for k in self.states:
            log_prob_x = log_sum(log_prob_x, self.log_pi[k] + self.log_e[k][x[0]] + b[0][k])
        return b, log_prob_x

    def posterior(self, x):
        """
        Input: a sequence x.
        Output: list[dict], for every position i and state k the probability P(zᵢ = k | x) = 2^(f[i][k] + b[i][k] − log2 P(x)) (a plain probability, not a log)        
        """
        f, log_px = self.forward(x)
        b, _ = self.backward(x)
        return [{k: 2 ** (f[i][k] + b[i][k] - log_px) for k in self.states} for i in range(len(x))]
 
    def posterior_path(self, x):
        """
        Input: a sequence x.
        Output: str, at every position the state with the highest posterior probability (ties: the first state).
        """
        path = ""
        for column in self.posterior(x):
            best = self.states[0]
            for k in self.states[1:]:
                if column[k] > column[best]:
                    best = k
            path += best
        return path

def task1(model):
    print(model.states)  # Output: ["H", "L"]
    print(model.alphabet)  # Output: ["s", "r"]
    print(model.joint_probability("sr", "HL"))  # Output: 0.054
    print(model.log_joint("sr", "HL"))  # Output: about -4.21
    print(log2(model.joint_probability("sr", "HL")))
    print(model.brute_force("ssrssrrrss")) # Output: ("HHHHHLLLHH", 5.53e-05, 0.000822)

def task2(model):
    # print(model.viterbi("ssrssrrrss"))
    print(model.viterbi("phchc"))
    print(model.vitebri_plain("phchc"))
    for record in fasta_seqs:
        seq = str(record.seq)
        print(f"Sequence length: {len(seq)}")
        print(f" log2: {model.viterbi(seq)[1]} \n plain: {model.vitebri_plain(seq)[1]}")

def task3(model):
    print(model.forward("ssrssrrrss"))
    print(model.backward("ssrssrrrss"))
    print(model.posterior("ssrssrrrss")[2]["L"])
    print(model.posterior_path("ssrssrrrss"))
    return

if __name__ == "__main__":
    model_weather = read_model("weather_model.json")
    model_toy = read_model("toy_model.json")
    fasta_seqs = list(SeqIO.parse("sequences.fasta", "fasta"))

    # task1(model_weather)
    # task2(model_weather)
    # task2(model_toy)
    task3(model_weather)

